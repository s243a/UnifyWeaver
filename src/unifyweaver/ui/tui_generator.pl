%! tui_generator.pl - Terminal User Interface generator from UI primitives
%
%  Generates ANSI-styled terminal output from declarative UI specifications.
%  Produces shell scripts that render forms and collect user input.
%
%  Example usage:
%      ?- generate_tui(layout(stack, [...], [...]), Code).
%      ?- generate_tui_script(ui_spec, "output.sh", Code).
%
%  @author UnifyWeaver
%  @version 1.0.0

:- module(tui_generator, [
    generate_tui/2,
    generate_tui/3,
    generate_tui_script/3,
    test_tui_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% ANSI ESCAPE CODE DEFINITIONS
% ============================================================================

% Reset
ansi_reset('\033[0m').

% Text styles
ansi_style(bold, '\033[1m').
ansi_style(dim, '\033[2m').
ansi_style(italic, '\033[3m').
ansi_style(underline, '\033[4m').
ansi_style(blink, '\033[5m').
ansi_style(reverse, '\033[7m').

% Foreground colors (basic)
ansi_fg(black, '\033[30m').
ansi_fg(red, '\033[31m').
ansi_fg(green, '\033[32m').
ansi_fg(yellow, '\033[33m').
ansi_fg(blue, '\033[34m').
ansi_fg(magenta, '\033[35m').
ansi_fg(cyan, '\033[36m').
ansi_fg(white, '\033[37m').
ansi_fg(default, '\033[39m').

% Bright foreground colors
ansi_fg(bright_black, '\033[90m').
ansi_fg(bright_red, '\033[91m').
ansi_fg(bright_green, '\033[92m').
ansi_fg(bright_yellow, '\033[93m').
ansi_fg(bright_blue, '\033[94m').
ansi_fg(bright_magenta, '\033[95m').
ansi_fg(bright_cyan, '\033[96m').
ansi_fg(bright_white, '\033[97m').

% Background colors
ansi_bg(black, '\033[40m').
ansi_bg(red, '\033[41m').
ansi_bg(green, '\033[42m').
ansi_bg(yellow, '\033[43m').
ansi_bg(blue, '\033[44m').
ansi_bg(magenta, '\033[45m').
ansi_bg(cyan, '\033[46m').
ansi_bg(white, '\033[47m').
ansi_bg(default, '\033[49m').

% Theme colors (matching the app theme)
theme_color(primary, '\033[38;5;204m').      % #e94560 - pinkish red
theme_color(secondary, '\033[38;5;75m').     % #0f3460 - dark blue
theme_color(background, '\033[48;5;234m').   % #1a1a2e - dark background
theme_color(surface, '\033[48;5;236m').      % #16213e - slightly lighter
theme_color(text, '\033[38;5;255m').         % white text
theme_color(text_dim, '\033[38;5;245m').     % dimmed text
theme_color(success, '\033[38;5;82m').       % green
theme_color(warning, '\033[38;5;214m').      % orange
theme_color(error, '\033[38;5;196m').        % red
theme_color(info, '\033[38;5;39m').          % blue

% Box drawing characters (Unicode)
box_char(top_left, '┌').
box_char(top_right, '┐').
box_char(bottom_left, '└').
box_char(bottom_right, '┘').
box_char(horizontal, '─').
box_char(vertical, '│').
box_char(t_down, '┬').
box_char(t_up, '┴').
box_char(t_right, '├').
box_char(t_left, '┤').
box_char(cross, '┼').

% Double-line box characters
box_char_double(top_left, '╔').
box_char_double(top_right, '╗').
box_char_double(bottom_left, '╚').
box_char_double(bottom_right, '╝').
box_char_double(horizontal, '═').
box_char_double(vertical, '║').

% ============================================================================
% MAIN ENTRY POINTS
% ============================================================================

%! generate_tui(+Spec, -Code) is det
%  Generate TUI output from a UI specification with default options.
generate_tui(Spec, Code) :-
    generate_tui(Spec, [], Code).

%! generate_tui(+Spec, +Options, -Code) is det
%  Generate TUI output from a UI specification with options.
generate_tui(Spec, Options, Code) :-
    get_option(indent, Options, Indent, 0),
    generate_node(Spec, Indent, Options, Code).

%! generate_tui_script(+Spec, +ScriptName, -Code) is det
%  Generate a complete shell script from a UI specification.
generate_tui_script(Spec, ScriptName, Code) :-
    generate_tui(Spec, [], BodyCode),
    ansi_reset(Reset),
    format(atom(Code), '#!/bin/bash
# Generated TUI script: ~w
# This script renders a terminal UI from declarative specifications

# Enable Unicode support
export LANG=en_US.UTF-8

# ANSI color definitions
RESET="~w"
BOLD="\\033[1m"
DIM="\\033[2m"

# Theme colors
PRIMARY="\\033[38;5;204m"
SECONDARY="\\033[38;5;75m"
BG="\\033[48;5;234m"
SURFACE="\\033[48;5;236m"
TEXT="\\033[38;5;255m"
TEXT_DIM="\\033[38;5;245m"
SUCCESS="\\033[38;5;82m"
WARNING="\\033[38;5;214m"
ERROR="\\033[38;5;196m"
INFO="\\033[38;5;39m"

# Clear screen and hide cursor
clear
tput civis

# Trap to restore cursor on exit
trap "tput cnorm; echo -e \\"$RESET\\"" EXIT

# Render UI
~w

# Show cursor
tput cnorm
', [ScriptName, Reset, BodyCode]).

% ============================================================================
% NODE GENERATION
% ============================================================================

%! generate_node(+Node, +Indent, +Options, -Code) is det
%  Generate TUI code for any node type.

% Layout nodes
generate_node(layout(Type, LayoutOpts, Children), Indent, Options, Code) :- !,
    generate_layout(Type, LayoutOpts, Children, Indent, Options, Code).

% Container nodes
generate_node(container(Type, ContainerOpts, Content), Indent, Options, Code) :- !,
    generate_container(Type, ContainerOpts, Content, Indent, Options, Code).

% Component nodes
generate_node(component(Type, CompOpts), Indent, Options, Code) :- !,
    generate_component(Type, CompOpts, Indent, Options, Code).

% Conditional nodes
generate_node(container(when, Condition, Content), Indent, Options, Code) :- !,
    generate_conditional(Condition, Content, Indent, Options, Code).

% Foreach nodes
generate_node(container(foreach, ForeachOpts, Template), Indent, Options, Code) :- !,
    generate_foreach(ForeachOpts, Template, Indent, Options, Code).

% Pattern expansion
generate_node(use_pattern(PatternName, Args), Indent, Options, Code) :- !,
    (expand_pattern(PatternName, Args, Expanded) ->
        generate_node(Expanded, Indent, Options, Code)
    ;
        indent_string(Indent, IndentStr),
        format(atom(Code), '~wecho "# Pattern: ~w"~n', [IndentStr, PatternName])
    ).

% Raw text
generate_node(text(Text), Indent, _Options, Code) :- !,
    indent_string(Indent, IndentStr),
    format(atom(Code), '~wecho "~w"~n', [IndentStr, Text]).

% Fallback
generate_node(Node, Indent, _Options, Code) :-
    indent_string(Indent, IndentStr),
    format(atom(Code), '~wecho "# Unknown node: ~w"~n', [IndentStr, Node]).

% ============================================================================
% LAYOUT GENERATION
% ============================================================================

%! generate_layout(+Type, +Options, +Children, +Indent, +GenOpts, -Code) is det

% Stack layout (vertical arrangement)
generate_layout(stack, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(spacing, Options, Spacing, 1),
    get_option(title, Options, Title, ''),
    indent_string(Indent, IndentStr),

    % Generate children with spacing
    generate_children_with_spacing(Children, Indent, GenOpts, Spacing, ChildrenCode),

    (Title \= '' ->
        theme_color(primary, Primary),
        ansi_reset(Reset),
        ansi_style(bold, Bold),
        format(atom(Code), '~wecho -e "~w~w~w~w"~n~w~wecho ""~n',
               [IndentStr, Primary, Bold, Title, Reset, ChildrenCode, IndentStr])
    ;
        Code = ChildrenCode
    ).

% Flex layout (horizontal arrangement)
generate_layout(flex, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(gap, Options, _Gap, 2),
    generate_horizontal_children(Children, Indent, GenOpts, Code).

% Grid layout
generate_layout(grid, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(columns, Options, Cols, 2),
    generate_grid_children(Children, Cols, Indent, GenOpts, Code).

% Center layout
generate_layout(center, _Options, Children, Indent, GenOpts, Code) :- !,
    generate_centered_children(Children, Indent, GenOpts, Code).

% Scroll layout (just render content in TUI)
generate_layout(scroll, _Options, Children, Indent, GenOpts, Code) :- !,
    generate_children(Children, Indent, GenOpts, Code).

% Wrap layout
generate_layout(wrap, _Options, Children, Indent, GenOpts, Code) :- !,
    generate_children(Children, Indent, GenOpts, Code).

% Default layout
generate_layout(_Type, _Options, Children, Indent, GenOpts, Code) :-
    generate_children(Children, Indent, GenOpts, Code).

% ============================================================================
% CONTAINER GENERATION
% ============================================================================

%! generate_container(+Type, +Options, +Content, +Indent, +GenOpts, -Code) is det

% Panel container with box border
generate_container(panel, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, ''),
    get_option(width, Options, Width, 60),
    indent_string(Indent, IndentStr),

    % Generate content
    NewIndent is Indent + 1,
    (is_list(Content) ->
        generate_children(Content, NewIndent, GenOpts, ContentCode)
    ;
        generate_node(Content, NewIndent, GenOpts, ContentCode)
    ),

    % Box drawing
    theme_color(surface, Surface),
    ansi_reset(Reset),

    generate_box_top(Title, Width, TopLine),
    generate_box_bottom(Width, BottomLine),

    format(atom(Code), '~wecho -e "~w~w~w"~n~w~wecho -e "~w~w~w"~n',
           [IndentStr, Surface, TopLine, Reset, ContentCode, IndentStr, Surface, BottomLine, Reset]).

% Card container
generate_container(card, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, ''),
    get_option(width, Options, Width, 50),
    indent_string(Indent, IndentStr),

    NewIndent is Indent + 1,
    (is_list(Content) ->
        generate_children(Content, NewIndent, GenOpts, ContentCode)
    ;
        generate_node(Content, NewIndent, GenOpts, ContentCode)
    ),

    theme_color(primary, Primary),
    ansi_style(bold, Bold),
    ansi_reset(Reset),

    generate_box_top_double(Title, Width, TopLine),
    generate_box_bottom_double(Width, BottomLine),

    format(atom(Code), '~wecho -e "~w~w~w~w"~n~w~wecho -e "~w"~n',
           [IndentStr, Primary, Bold, TopLine, Reset, ContentCode, IndentStr, BottomLine]).

% Section container
generate_container(section, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, 'Section'),
    indent_string(Indent, IndentStr),

    NewIndent is Indent + 1,
    (is_list(Content) ->
        generate_children(Content, NewIndent, GenOpts, ContentCode)
    ;
        generate_node(Content, NewIndent, GenOpts, ContentCode)
    ),

    theme_color(primary, Primary),
    ansi_style(bold, Bold),
    ansi_style(underline, Underline),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w~w~w~w"~n~wecho ""~n~w',
           [IndentStr, Primary, Bold, Underline, Title, Reset, IndentStr, ContentCode]).

% Modal container
generate_container(modal, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, 'Dialog'),
    get_option(width, Options, Width, 50),
    indent_string(Indent, IndentStr),

    NewIndent is Indent + 1,
    (is_list(Content) ->
        generate_children(Content, NewIndent, GenOpts, ContentCode)
    ;
        generate_node(Content, NewIndent, GenOpts, ContentCode)
    ),

    theme_color(background, Bg),
    theme_color(primary, Primary),
    ansi_style(bold, Bold),
    ansi_reset(Reset),

    generate_box_top_double(Title, Width, TopLine),
    generate_box_bottom_double(Width, BottomLine),

    format(atom(Code), '~w# Modal: ~w~n~wecho -e "~w~w~w~w~w"~n~w~wecho -e "~w"~n',
           [IndentStr, Title, IndentStr, Bg, Primary, Bold, TopLine, Reset, ContentCode, IndentStr, BottomLine]).

% Conditional container
generate_container(when, Condition, Content, Indent, GenOpts, Code) :- !,
    generate_conditional(Condition, Content, Indent, GenOpts, Code).

% Unless container
generate_container(unless, Condition, Content, Indent, GenOpts, Code) :- !,
    generate_conditional(not(Condition), Content, Indent, GenOpts, Code).

% Default container
generate_container(_Type, _Options, Content, Indent, GenOpts, Code) :-
    (is_list(Content) ->
        generate_children(Content, Indent, GenOpts, Code)
    ;
        generate_node(Content, Indent, GenOpts, Code)
    ).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%! generate_component(+Type, +Options, +Indent, +GenOpts, -Code) is det

% Text component
generate_component(text, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    get_option(style, Options, Style, normal),
    indent_string(Indent, IndentStr),

    text_style_codes(Style, StyleCode),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w~w"~n', [IndentStr, StyleCode, Content, Reset]).

% Heading component
generate_component(heading, Options, Indent, _GenOpts, Code) :- !,
    get_option(level, Options, Level, 1),
    get_option(content, Options, Content, ''),
    indent_string(Indent, IndentStr),

    heading_style(Level, StyleCode),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w~w"~n~wecho ""~n',
           [IndentStr, StyleCode, Content, Reset, IndentStr]).

% Label component
generate_component(label, Options, Indent, _GenOpts, Code) :- !,
    get_option(text, Options, Text, ''),
    indent_string(Indent, IndentStr),

    theme_color(text_dim, Dim),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w~w"~n', [IndentStr, Dim, Text, Reset]).

% Button component
generate_component(button, Options, Indent, _GenOpts, Code) :- !,
    get_option_label(Options, Label),
    get_option(variant, Options, Variant, primary),
    get_option(on_click, Options, OnClick, ''),
    indent_string(Indent, IndentStr),

    button_style(Variant, StyleCode),
    ansi_reset(Reset),
    label_to_shell(Label, LabelCode),

    (OnClick \= '' ->
        format(atom(Code), '~wecho -e "~w [ ~w ] ~w"  # on_click: ~w~n',
               [IndentStr, StyleCode, LabelCode, Reset, OnClick])
    ;
        format(atom(Code), '~wecho -e "~w [ ~w ] ~w"~n',
               [IndentStr, StyleCode, LabelCode, Reset])
    ).

% Text input component
generate_component(text_input, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, ''),
    get_option(bind, Options, Bind, 'input'),
    get_option(placeholder, Options, Placeholder, ''),
    get_option(type, Options, Type, text),
    indent_string(Indent, IndentStr),

    theme_color(text_dim, Dim),
    theme_color(text, Text),
    ansi_reset(Reset),

    (Type == password ->
        ReadOpts = '-s '
    ;
        ReadOpts = ''
    ),

    (Label \= '' ->
        format(atom(Code), '~wecho -e "~w~w:~w"~n~wread ~w-p "~w> ~w" ~w~n~wecho ""~n',
               [IndentStr, Dim, Label, Reset, IndentStr, ReadOpts, Text, Reset, Bind, IndentStr])
    ;
        (Placeholder \= '' ->
            format(atom(Code), '~wread ~w-p "~w~w~w> ~w" ~w~n',
                   [IndentStr, ReadOpts, Dim, Placeholder, Reset, Text, Bind])
        ;
            format(atom(Code), '~wread ~w-p "> " ~w~n', [IndentStr, ReadOpts, Bind])
        )
    ).

% Textarea component
generate_component(textarea, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, ''),
    get_option(bind, Options, Bind, 'text'),
    get_option(rows, Options, Rows, 4),
    get_option(placeholder, Options, _Placeholder, ''),  % Placeholder not shown in TUI text area
    indent_string(Indent, IndentStr),

    theme_color(text_dim, Dim),
    theme_color(text, Text),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w~w (Enter ~w lines, empty line to finish):"~n~w~w=""~n~wfor i in $(seq 1 ~w); do~n~w  read -p "~w> ~w" line~n~w  [ -z "$line" ] && break~n~w  ~w="$~w$line\\n"~n~wdone~n',
           [IndentStr, Dim, Label, Reset, Rows, IndentStr, Bind, IndentStr, Rows,
            IndentStr, Text, Reset, IndentStr, IndentStr, Bind, Bind, IndentStr]).

% Checkbox component
generate_component(checkbox, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, ''),
    get_option(bind, Options, Bind, 'checked'),
    indent_string(Indent, IndentStr),

    theme_color(primary, Primary),
    ansi_reset(Reset),

    format(atom(Code), '~wread -p "~w~w~w [y/N]: " ~w_input~n~w[ "$~w_input" = "y" ] || [ "$~w_input" = "Y" ] && ~w=true || ~w=false~n',
           [IndentStr, Primary, Label, Reset, Bind, IndentStr, Bind, Bind, Bind, Bind]).

% Select component
generate_component(select, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, ''),
    get_option(bind, Options, Bind, 'selected'),
    get_option(options, Options, SelectOptions, []),
    indent_string(Indent, IndentStr),

    theme_color(text_dim, Dim),
    theme_color(primary, Primary),
    theme_color(text, Text),
    ansi_reset(Reset),

    generate_select_options(SelectOptions, 1, Dim, Primary, Reset, OptionsCode),
    length(SelectOptions, NumOptions),

    format(atom(Code), '~wecho -e "~w~w:~w"~n~w~wecho -e "~wSelect (1-~w):~w"~n~wread -p "> " ~w~n',
           [IndentStr, Dim, Label, Reset, OptionsCode, IndentStr, Text, NumOptions, Reset, IndentStr, Bind]).

% Switch component
generate_component(switch, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, ''),
    get_option(bind, Options, Bind, 'enabled'),
    indent_string(Indent, IndentStr),

    theme_color(primary, Primary),
    theme_color(success, Success),
    ansi_reset(Reset),

    format(atom(Code), '~wread -p "~w~w~w [on/OFF]: " ~w_input~n~w[ "$~w_input" = "on" ] && ~w=true && echo -e "~w◉ ON~w" || ~w=false && echo -e "○ OFF"~n',
           [IndentStr, Primary, Label, Reset, Bind, IndentStr, Bind, Bind, Success, Reset, Bind]).

% Spinner component
generate_component(spinner, Options, Indent, _GenOpts, Code) :- !,
    get_option(text, Options, Text, 'Loading...'),
    indent_string(Indent, IndentStr),

    theme_color(primary, Primary),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w⠋ ~w~w"~n', [IndentStr, Primary, Text, Reset]).

% Progress component
generate_component(progress, Options, Indent, _GenOpts, Code) :- !,
    get_option(value, Options, Value, 50),
    get_option(max, Options, Max, 100),
    get_option(width, Options, Width, 40),
    indent_string(Indent, IndentStr),

    theme_color(primary, Primary),
    theme_color(text_dim, Dim),
    ansi_reset(Reset),

    Percent is (Value * 100) // Max,
    Filled is (Value * Width) // Max,
    Empty is Width - Filled,

    format(atom(Code), '~wecho -e "~w~w~w~w ~w%~w"~n',
           [IndentStr, Primary, filled_bar(Filled), Dim, empty_bar(Empty), Percent, Reset]).

% Badge component
generate_component(badge, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    get_option(variant, Options, Variant, info),
    indent_string(Indent, IndentStr),

    badge_style(Variant, StyleCode),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w ~w ~w"~n', [IndentStr, StyleCode, Content, Reset]).

% Divider component
generate_component(divider, _Options, Indent, _GenOpts, Code) :- !,
    indent_string(Indent, IndentStr),
    theme_color(text_dim, Dim),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w────────────────────────────────────────~w"~n', [IndentStr, Dim, Reset]).

% Spacer component
generate_component(spacer, Options, Indent, _GenOpts, Code) :- !,
    get_option(height, Options, Height, 1),
    indent_string(Indent, IndentStr),

    generate_empty_lines(Height, IndentStr, Code).

% Image component (show placeholder in TUI)
generate_component(image, Options, Indent, _GenOpts, Code) :- !,
    get_option(alt, Options, Alt, 'Image'),
    get_option(src, Options, Src, ''),
    indent_string(Indent, IndentStr),

    theme_color(text_dim, Dim),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w[Image: ~w]~w"  # src: ~w~n', [IndentStr, Dim, Alt, Reset, Src]).

% Icon component
generate_component(icon, Options, Indent, _GenOpts, Code) :- !,
    get_option(name, Options, Name, 'star'),
    indent_string(Indent, IndentStr),

    icon_to_unicode(Name, Unicode),
    theme_color(primary, Primary),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w~w"~n', [IndentStr, Primary, Unicode, Reset]).

% Avatar component
generate_component(avatar, Options, Indent, _GenOpts, Code) :- !,
    get_option(name, Options, Name, '?'),
    indent_string(Indent, IndentStr),

    (atom_length(Name, Len), Len > 0 ->
        atom_chars(Name, [Initial|_])
    ;
        Initial = '?'
    ),

    theme_color(primary, Primary),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w(~w)~w"~n', [IndentStr, Primary, Initial, Reset]).

% Link component
generate_component(link, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, 'Link'),
    get_option(href, Options, Href, ''),
    indent_string(Indent, IndentStr),

    theme_color(info, Info),
    ansi_style(underline, Underline),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w~w~w"  # href: ~w~n',
           [IndentStr, Info, Underline, Label, Reset, Href]).

% Tabs component
generate_component(tabs, Options, Indent, _GenOpts, Code) :- !,
    get_option(items, Options, Items, []),
    get_option(active, Options, _Active, 0),
    indent_string(Indent, IndentStr),

    theme_color(primary, Primary),
    theme_color(text_dim, Dim),
    ansi_style(bold, Bold),
    ansi_reset(Reset),

    generate_tab_items(Items, 0, Primary, Bold, Dim, Reset, TabsCode),

    format(atom(Code), '~wecho -e "~w"~n~wecho ""~n', [IndentStr, TabsCode, IndentStr]).

% Code/pre component
generate_component(code, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    indent_string(Indent, IndentStr),

    theme_color(surface, Surface),
    theme_color(text, Text),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w ~w ~w"~n', [IndentStr, Surface, Text, Content, Reset]).

generate_component(pre, Options, Indent, GenOpts, Code) :- !,
    generate_component(code, Options, Indent, GenOpts, Code).

% Alert component
generate_component(alert, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    get_option(variant, Options, Variant, info),
    indent_string(Indent, IndentStr),

    alert_style(Variant, StyleCode, Icon),
    ansi_reset(Reset),

    format(atom(Code), '~wecho -e "~w~w ~w~w"~n', [IndentStr, StyleCode, Icon, Content, Reset]).

% Default component fallback
generate_component(Type, Options, Indent, _GenOpts, Code) :-
    indent_string(Indent, IndentStr),
    format(atom(Code), '~wecho "# Component: ~w ~w"~n', [IndentStr, Type, Options]).

% ============================================================================
% CONDITIONAL GENERATION
% ============================================================================

generate_conditional(Condition, Content, Indent, GenOpts, Code) :-
    condition_to_shell(Condition, ShellCond),
    NewIndent is Indent + 1,
    (is_list(Content) ->
        generate_children(Content, NewIndent, GenOpts, ContentCode)
    ;
        generate_node(Content, NewIndent, GenOpts, ContentCode)
    ),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~wif ~w; then~n~w~wfi~n', [IndentStr, ShellCond, ContentCode, IndentStr]).

% ============================================================================
% FOREACH GENERATION
% ============================================================================

generate_foreach(ForeachOpts, Template, Indent, GenOpts, Code) :-
    get_option(items, ForeachOpts, Items, []),
    get_option(as, ForeachOpts, ItemVar, 'item'),
    indent_string(Indent, IndentStr),

    NewIndent is Indent + 1,
    generate_node(Template, NewIndent, GenOpts, TemplateCode),

    (is_binding(Items) ->
        binding_to_shell(Items, ShellItems),
        format(atom(Code), '~wfor ~w in ~w; do~n~w~wdone~n',
               [IndentStr, ItemVar, ShellItems, TemplateCode, IndentStr])
    ;
        generate_static_foreach(Items, ItemVar, Template, Indent, GenOpts, Code)
    ).

generate_static_foreach([], _ItemVar, _Template, _Indent, _GenOpts, '') :- !.
generate_static_foreach([Item|Rest], ItemVar, Template, Indent, GenOpts, Code) :-
    indent_string(Indent, IndentStr),
    generate_node(Template, Indent, GenOpts, TemplateCode),
    generate_static_foreach(Rest, ItemVar, Template, Indent, GenOpts, RestCode),
    format(atom(Code), '~w~w="~w"~n~w~w', [IndentStr, ItemVar, Item, TemplateCode, RestCode]).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

%! generate_children(+Children, +Indent, +Options, -Code) is det
generate_children([], _Indent, _Options, '') :- !.
generate_children([Child|Rest], Indent, Options, Code) :-
    generate_node(Child, Indent, Options, ChildCode),
    generate_children(Rest, Indent, Options, RestCode),
    atom_concat(ChildCode, RestCode, Code).

%! generate_children_with_spacing(+Children, +Indent, +Options, +Spacing, -Code) is det
generate_children_with_spacing([], _Indent, _Options, _Spacing, '') :- !.
generate_children_with_spacing([Child], Indent, Options, _Spacing, Code) :- !,
    generate_node(Child, Indent, Options, Code).
generate_children_with_spacing([Child|Rest], Indent, Options, Spacing, Code) :-
    generate_node(Child, Indent, Options, ChildCode),
    indent_string(Indent, IndentStr),
    generate_empty_lines(Spacing, IndentStr, SpacingCode),
    generate_children_with_spacing(Rest, Indent, Options, Spacing, RestCode),
    format(atom(Code), '~w~w~w', [ChildCode, SpacingCode, RestCode]).

%! generate_horizontal_children(+Children, +Indent, +Options, -Code) is det
generate_horizontal_children(Children, Indent, _Options, Code) :-
    indent_string(Indent, IndentStr),
    generate_horizontal_items(Children, Items),
    format(atom(Code), '~wecho -e "~w"~n', [IndentStr, Items]).

generate_horizontal_items([], '') :- !.
generate_horizontal_items([component(text, Opts)|Rest], Code) :- !,
    get_option(content, Opts, Content, ''),
    generate_horizontal_items(Rest, RestCode),
    format(atom(Code), '~w  ~w', [Content, RestCode]).
generate_horizontal_items([component(button, Opts)|Rest], Code) :- !,
    get_option_label(Opts, Label),
    label_to_shell(Label, LabelCode),
    generate_horizontal_items(Rest, RestCode),
    format(atom(Code), '[ ~w ]  ~w', [LabelCode, RestCode]).
generate_horizontal_items([_|Rest], Code) :-
    generate_horizontal_items(Rest, Code).

%! generate_grid_children(+Children, +Cols, +Indent, +Options, -Code) is det
generate_grid_children(Children, Cols, Indent, Options, Code) :-
    chunk_list(Children, Cols, Rows),
    generate_grid_rows(Rows, Indent, Options, Code).

generate_grid_rows([], _Indent, _Options, '') :- !.
generate_grid_rows([Row|Rest], Indent, Options, Code) :-
    generate_horizontal_children(Row, Indent, Options, RowCode),
    generate_grid_rows(Rest, Indent, Options, RestCode),
    atom_concat(RowCode, RestCode, Code).

%! generate_centered_children(+Children, +Indent, +Options, -Code) is det
generate_centered_children(Children, Indent, Options, Code) :-
    indent_string(Indent, IndentStr),
    generate_children(Children, Indent, Options, ChildrenCode),
    % Add centering comment (actual centering requires terminal width calculation)
    format(atom(Code), '~w# Centered content~n~w', [IndentStr, ChildrenCode]).

%! chunk_list(+List, +Size, -Chunks) is det
chunk_list([], _Size, []) :- !.
chunk_list(List, Size, [Chunk|Rest]) :-
    length(Chunk, Size),
    append(Chunk, Remaining, List), !,
    chunk_list(Remaining, Size, Rest).
chunk_list(List, _Size, [List]).

%! generate_empty_lines(+N, +Indent, -Code) is det
generate_empty_lines(0, _Indent, '') :- !.
generate_empty_lines(N, Indent, Code) :-
    N > 0,
    N1 is N - 1,
    generate_empty_lines(N1, Indent, RestCode),
    format(atom(Code), '~wecho ""~n~w', [Indent, RestCode]).

%! indent_string(+N, -Str) is det
indent_string(0, '') :- !.
indent_string(N, Str) :-
    N > 0,
    N1 is N - 1,
    indent_string(N1, Rest),
    atom_concat('  ', Rest, Str).

%! get_option(+Key, +Options, -Value, +Default) is det
get_option(Key, Options, Value, _Default) :-
    Term =.. [Key, Value],
    member(Term, Options), !.
get_option(_Key, _Options, Default, Default).

%! get_option_label(+Options, -Label) is det
get_option_label(Options, conditional(Cond, TrueVal, FalseVal)) :-
    member(label(Cond, TrueVal, FalseVal), Options), !.
get_option_label(Options, simple(Label)) :-
    member(label(Label), Options), !.
get_option_label(_, simple('Button')).

%! label_to_shell(+Label, -Code) is det
label_to_shell(simple(Text), Text) :- !.
label_to_shell(conditional(Cond, TrueVal, FalseVal), Code) :- !,
    condition_to_shell(Cond, ShellCond),
    format(atom(Code), '$( ~w && echo "~w" || echo "~w" )', [ShellCond, TrueVal, FalseVal]).
label_to_shell(Text, Text).

% ============================================================================
% CONDITION CONVERSION
% ============================================================================

condition_to_shell(var(X), Code) :- !,
    format(atom(Code), '[ -n "$~w" ]', [X]).
condition_to_shell(not(C), Code) :- !,
    condition_to_shell(C, Inner),
    format(atom(Code), '! ~w', [Inner]).
condition_to_shell(and(A, B), Code) :- !,
    condition_to_shell(A, CA),
    condition_to_shell(B, CB),
    format(atom(Code), '~w && ~w', [CA, CB]).
condition_to_shell(or(A, B), Code) :- !,
    condition_to_shell(A, CA),
    condition_to_shell(B, CB),
    format(atom(Code), '( ~w || ~w )', [CA, CB]).
condition_to_shell(eq(A, B), Code) :- !,
    value_to_shell(A, VA),
    value_to_shell(B, VB),
    format(atom(Code), '[ "~w" = "~w" ]', [VA, VB]).
condition_to_shell(not_eq(A, B), Code) :- !,
    value_to_shell(A, VA),
    value_to_shell(B, VB),
    format(atom(Code), '[ "~w" != "~w" ]', [VA, VB]).
condition_to_shell(empty(C), Code) :- !,
    value_to_shell(C, VC),
    format(atom(Code), '[ -z "~w" ]', [VC]).
condition_to_shell(true, 'true') :- !.
condition_to_shell(false, 'false') :- !.
condition_to_shell(Atom, Code) :-
    atom(Atom), !,
    format(atom(Code), '[ -n "$~w" ]', [Atom]).
condition_to_shell(Term, Code) :-
    format(atom(Code), '~w', [Term]).

value_to_shell(var(X), Code) :- !,
    format(atom(Code), '$~w', [X]).
value_to_shell(X, X).

%! binding_to_shell(+Binding, -Code) is det
binding_to_shell(var(X), Code) :- !,
    format(atom(Code), '"$~w"', [X]).
binding_to_shell(X, Code) :-
    format(atom(Code), '"~w"', [X]).

%! is_binding(+Term) is nondet
is_binding(var(_)).
is_binding(bind(_)).

% ============================================================================
% STYLE HELPERS
% ============================================================================

%! text_style_codes(+Style, -Code) is det
text_style_codes(normal, Code) :- theme_color(text, Code).
text_style_codes(bold, Code) :-
    theme_color(text, Text),
    ansi_style(bold, Bold),
    atom_concat(Text, Bold, Code).
text_style_codes(dim, Code) :- theme_color(text_dim, Code).
text_style_codes(header, Code) :-
    theme_color(primary, Primary),
    ansi_style(bold, Bold),
    atom_concat(Primary, Bold, Code).
text_style_codes(_, Code) :- theme_color(text, Code).

%! heading_style(+Level, -Code) is det
heading_style(1, Code) :-
    theme_color(primary, Primary),
    ansi_style(bold, Bold),
    atom_concat(Primary, Bold, Code).
heading_style(2, Code) :-
    theme_color(primary, Primary),
    ansi_style(bold, Bold),
    atom_concat(Primary, Bold, Code).
heading_style(3, Code) :-
    theme_color(text, Text),
    ansi_style(bold, Bold),
    atom_concat(Text, Bold, Code).
heading_style(_, Code) :- theme_color(text, Code).

%! button_style(+Variant, -Code) is det
button_style(primary, Code) :-
    theme_color(primary, Fg),
    ansi_style(bold, Bold),
    atom_concat(Fg, Bold, Code).
button_style(secondary, Code) :- theme_color(text_dim, Code).
button_style(danger, Code) :- theme_color(error, Code).
button_style(success, Code) :- theme_color(success, Code).
button_style(_, Code) :- theme_color(primary, Code).

%! badge_style(+Variant, -Code) is det
badge_style(info, Code) :- theme_color(info, Code).
badge_style(success, Code) :- theme_color(success, Code).
badge_style(warning, Code) :- theme_color(warning, Code).
badge_style(error, Code) :- theme_color(error, Code).
badge_style(_, Code) :- theme_color(text_dim, Code).

%! alert_style(+Variant, -StyleCode, -Icon) is det
alert_style(info, Code, 'ℹ') :- theme_color(info, Code).
alert_style(success, Code, '✓') :- theme_color(success, Code).
alert_style(warning, Code, '⚠') :- theme_color(warning, Code).
alert_style(error, Code, '✗') :- theme_color(error, Code).
alert_style(_, Code, '•') :- theme_color(text, Code).

%! icon_to_unicode(+Name, -Unicode) is det
icon_to_unicode(star, '★').
icon_to_unicode(check, '✓').
icon_to_unicode(cross, '✗').
icon_to_unicode(arrow_right, '→').
icon_to_unicode(arrow_left, '←').
icon_to_unicode(arrow_up, '↑').
icon_to_unicode(arrow_down, '↓').
icon_to_unicode(folder, '📁').
icon_to_unicode(file, '📄').
icon_to_unicode(search, '🔍').
icon_to_unicode(settings, '⚙').
icon_to_unicode(user, '👤').
icon_to_unicode(home, '🏠').
icon_to_unicode(edit, '✎').
icon_to_unicode(delete, '🗑').
icon_to_unicode(refresh, '↻').
icon_to_unicode(info, 'ℹ').
icon_to_unicode(warning, '⚠').
icon_to_unicode(error, '⊘').
icon_to_unicode(_, '•').

% ============================================================================
% BOX DRAWING HELPERS
% ============================================================================

%! generate_box_top(+Title, +Width, -Line) is det
generate_box_top('', Width, Line) :- !,
    box_char(top_left, TL),
    box_char(top_right, TR),
    box_char(horizontal, H),
    W is Width - 2,
    repeat_char(H, W, Middle),
    format(atom(Line), '~w~w~w', [TL, Middle, TR]).
generate_box_top(Title, Width, Line) :-
    box_char(top_left, TL),
    box_char(top_right, TR),
    box_char(horizontal, H),
    atom_length(Title, TitleLen),
    LeftPad is 2,
    RightLen is Width - TitleLen - LeftPad - 4,
    (RightLen > 0 -> RightPad = RightLen ; RightPad = 1),
    repeat_char(H, LeftPad, Left),
    repeat_char(H, RightPad, Right),
    format(atom(Line), '~w~w ~w ~w~w', [TL, Left, Title, Right, TR]).

%! generate_box_bottom(+Width, -Line) is det
generate_box_bottom(Width, Line) :-
    box_char(bottom_left, BL),
    box_char(bottom_right, BR),
    box_char(horizontal, H),
    W is Width - 2,
    repeat_char(H, W, Middle),
    format(atom(Line), '~w~w~w', [BL, Middle, BR]).

%! generate_box_top_double(+Title, +Width, -Line) is det
generate_box_top_double('', Width, Line) :- !,
    box_char_double(top_left, TL),
    box_char_double(top_right, TR),
    box_char_double(horizontal, H),
    W is Width - 2,
    repeat_char(H, W, Middle),
    format(atom(Line), '~w~w~w', [TL, Middle, TR]).
generate_box_top_double(Title, Width, Line) :-
    box_char_double(top_left, TL),
    box_char_double(top_right, TR),
    box_char_double(horizontal, H),
    atom_length(Title, TitleLen),
    LeftPad is 2,
    RightLen is Width - TitleLen - LeftPad - 4,
    (RightLen > 0 -> RightPad = RightLen ; RightPad = 1),
    repeat_char(H, LeftPad, Left),
    repeat_char(H, RightPad, Right),
    format(atom(Line), '~w~w ~w ~w~w', [TL, Left, Title, Right, TR]).

%! generate_box_bottom_double(+Width, -Line) is det
generate_box_bottom_double(Width, Line) :-
    box_char_double(bottom_left, BL),
    box_char_double(bottom_right, BR),
    box_char_double(horizontal, H),
    W is Width - 2,
    repeat_char(H, W, Middle),
    format(atom(Line), '~w~w~w', [BL, Middle, BR]).

%! repeat_char(+Char, +N, -Result) is det
repeat_char(_Char, N, '') :- N =< 0, !.
repeat_char(Char, N, Result) :-
    N > 0,
    N1 is N - 1,
    repeat_char(Char, N1, Rest),
    atom_concat(Char, Rest, Result).

%! generate_select_options(+Options, +Index, +Dim, +Primary, +Reset, -Code) is det
generate_select_options([], _Index, _Dim, _Primary, _Reset, '') :- !.
generate_select_options([Opt|Rest], Index, Dim, Primary, Reset, Code) :-
    (Opt = option(_Value, Label) ->
        true  % Value used for binding, Label for display
    ;
        Label = Opt
    ),
    NextIndex is Index + 1,
    generate_select_options(Rest, NextIndex, Dim, Primary, Reset, RestCode),
    format(atom(Code), 'echo -e "  ~w~w.~w ~w~w"~n~w',
           [Primary, Index, Reset, Label, Dim, RestCode]).

%! generate_tab_items(+Items, +ActiveIndex, +Primary, +Bold, +Dim, +Reset, -Code) is det
generate_tab_items([], _Index, _Primary, _Bold, _Dim, _Reset, '') :- !.
generate_tab_items([Item|Rest], Index, Primary, Bold, Dim, Reset, Code) :-
    NextIndex is Index + 1,
    generate_tab_items(Rest, NextIndex, Primary, Bold, Dim, Reset, RestCode),
    (Index == 0 ->
        format(atom(Code), '~w~w[~w]~w  ~w', [Primary, Bold, Item, Reset, RestCode])
    ;
        format(atom(Code), '~w ~w ~w  ~w', [Dim, Item, Reset, RestCode])
    ).

% Pattern expansion (stub - actual implementation would use ui_patterns module)
expand_pattern(_Name, _Args, _Expanded) :- fail.

% ============================================================================
% TESTS
% ============================================================================

test_tui_generator :-
    format('~n=== TUI Generator Tests ===~n~n', []),

    % Test 1: Simple text component
    format('Test 1: Simple text component...~n', []),
    generate_tui(component(text, [content('Hello World')]), T1),
    format('  Output: ~w~n', [T1]),

    % Test 2: Button component
    format('Test 2: Button component...~n', []),
    generate_tui(component(button, [label('Click Me'), on_click(handleClick)]), T2),
    format('  Output: ~w~n', [T2]),

    % Test 3: Stack layout
    format('Test 3: Stack layout...~n', []),
    generate_tui(layout(stack, [spacing(1)], [
        component(heading, [level(1), content('Title')]),
        component(text, [content('Description')])
    ]), T3),
    format('  Output:~n~w~n', [T3]),

    % Test 4: Form with inputs
    format('Test 4: Form with inputs...~n', []),
    generate_tui(layout(stack, [spacing(1)], [
        component(text_input, [label('Email'), bind(email), placeholder('Enter email')]),
        component(text_input, [label('Password'), type(password), bind(password)]),
        component(button, [label('Submit'), on_click(submit)])
    ]), T4),
    format('  Output:~n~w~n', [T4]),

    % Test 5: Conditional rendering
    format('Test 5: Conditional rendering...~n', []),
    generate_tui(container(when, var(isLoggedIn), component(text, [content('Welcome!')])), T5),
    format('  Output: ~w~n', [T5]),

    % Test 6: Panel container
    format('Test 6: Panel container...~n', []),
    generate_tui(container(panel, [title('My Panel'), width(40)], [
        component(text, [content('Panel content')])
    ]), T6),
    format('  Output:~n~w~n', [T6]),

    % Test 7: Select component
    format('Test 7: Select component...~n', []),
    generate_tui(component(select, [label('Choose'), bind(choice), options([option(a, 'Option A'), option(b, 'Option B')])]), T7),
    format('  Output:~n~w~n', [T7]),

    % Test 8: Full script generation
    format('Test 8: Full script generation...~n', []),
    generate_tui_script(layout(stack, [], [
        component(heading, [level(1), content('My App')]),
        component(button, [label('Click'), on_click(handleClick)])
    ]), 'test.sh', T8),
    format('  Output (first 500 chars):~n~w~n', [T8]),

    format('~n=== Tests Complete ===~n', []).

:- format('TUI Generator module loaded~n', []).
