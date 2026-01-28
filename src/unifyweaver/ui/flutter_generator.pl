% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% flutter_generator.pl - Generate Flutter/Dart code from UI primitives
%
% Compiles declarative UI specifications to Flutter widgets using Dart syntax.
%
% Usage:
%   use_module('src/unifyweaver/ui/flutter_generator').
%   UI = layout(stack, [spacing(16)], [
%       component(text, [content("Hello")]),
%       component(button, [label("Click"), on_click(submit)])
%   ]),
%   generate_flutter_widget(UI, Widget).

:- module(flutter_generator, [
    % Widget generation
    generate_flutter_widget/2,      % generate_flutter_widget(+UISpec, -Widget)
    generate_flutter_widget/3,      % generate_flutter_widget(+UISpec, +Options, -Widget)

    % Full file generation
    generate_flutter_file/3,        % generate_flutter_file(+UISpec, +Options, -File)

    % Testing
    test_flutter_generator/0
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

% Load pattern system (optional - patterns expanded on-the-fly if available)
:- catch(use_module('ui_patterns'), _, true).

% ============================================================================
% WIDGET GENERATION
% ============================================================================

%! generate_flutter_widget(+UISpec, -Widget) is det
%  Generate Flutter widget from UI specification.
generate_flutter_widget(UISpec, Widget) :-
    generate_flutter_widget(UISpec, [], Widget).

%! generate_flutter_widget(+UISpec, +Options, -Widget) is det
%  Generate Flutter widget with options.
%  Options:
%    indent(N) - Starting indentation level
generate_flutter_widget(UISpec, Options, Widget) :-
    get_option(indent, Options, Indent, 0),
    generate_node(UISpec, Indent, Options, Widget).

% ============================================================================
% NODE GENERATION
% ============================================================================

%! generate_node(+Node, +Indent, +Options, -Code) is det
%  Generate Flutter code for any UI node.

% Layout nodes
generate_node(layout(Type, LayoutOpts, Children), Indent, Options, Code) :- !,
    generate_layout(Type, LayoutOpts, Children, Indent, Options, Code).

% Container nodes
generate_node(container(Type, ContainerOpts, Content), Indent, Options, Code) :- !,
    generate_container(Type, ContainerOpts, Content, Indent, Options, Code).

% Component nodes
generate_node(component(Type, CompOpts), Indent, Options, Code) :- !,
    generate_component(Type, CompOpts, Indent, Options, Code).

% Pattern nodes - expand and generate
generate_node(use_pattern(Name, Args), Indent, Options, Code) :-
    (   current_predicate(ui_patterns:expand_pattern/3)
    ->  ui_patterns:expand_pattern(Name, Args, Expanded),
        generate_node(Expanded, Indent, Options, Code)
    ;   indent_string(Indent, IndentStr),
        format(atom(Code), '~w// Pattern ~w not expanded: ui_patterns module not loaded~n', [IndentStr, Name])
    ), !.

% Conditional: when
generate_node(when(Condition, Content), Indent, Options, Code) :- !,
    generate_conditional(Condition, Content, Indent, Options, Code).

% Conditional: unless
generate_node(unless(Condition, Content), Indent, Options, Code) :- !,
    condition_to_dart(Condition, DartCond),
    generate_node(Content, Indent, Options, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~wif (!(~w)) ~w', [IndentStr, DartCond, ContentCode]).

% Foreach iteration - generates a Column wrapping mapped items
generate_node(foreach(Items, Var, Template), Indent, Options, Code) :- !,
    items_to_dart(Items, DartItems),
    NextIndent is Indent + 2,
    % Track loop variable so dotted_to_dart doesn't add underscore prefix
    get_option(loop_vars, Options, ExistingLoopVars, []),
    NewOptions = [loop_vars([Var|ExistingLoopVars])|Options],
    generate_node(Template, NextIndent, NewOptions, TemplateCode),
    indent_string(Indent, IndentStr),
    indent_string(Indent + 1, NextIndentStr),
    % Wrap in Column for proper widget tree structure
    format(atom(Code), '~wColumn(~n~wchildren: ~w.map((~w) => ~w).toList(),~n~w),~n',
           [IndentStr, NextIndentStr, DartItems, Var, TemplateCode, IndentStr]).

% List of nodes
generate_node([], _Indent, _Options, '') :- !.
generate_node([H|T], Indent, Options, Code) :- !,
    generate_node(H, Indent, Options, HCode),
    generate_node(T, Indent, Options, TCode),
    atom_concat(HCode, TCode, Code).

% Fallback
generate_node(Node, Indent, _Options, Code) :-
    indent_string(Indent, IndentStr),
    functor(Node, Functor, _),
    format(atom(Code), '~w// Unknown node: ~w~n', [IndentStr, Functor]).

% ============================================================================
% LAYOUT GENERATION
% ============================================================================

%! generate_layout(+Type, +Options, +Children, +Indent, +GenOpts, -Code) is det

% Stack layout -> Column or Row
generate_layout(stack, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(direction, Options, Dir, column),
    get_option(spacing, Options, Spacing, 0),
    get_option(align, Options, Align, stretch),
    get_option(justify, Options, Justify, start),

    (Dir = column -> Widget = 'Column' ; Widget = 'Row'),
    main_axis_alignment(Justify, MainAxis),
    cross_axis_alignment(Align, CrossAxis),

    generate_children_with_spacing(Children, Spacing, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    NextIndent is Indent + 1,
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~w~w(~n~wmainAxisAlignment: ~w,~n~wcrossAxisAlignment: ~w,~n~wchildren: [~n~w~w],~n~w),~n',
           [IndentStr, Widget, NextIndentStr, MainAxis, NextIndentStr, CrossAxis,
            NextIndentStr, ChildrenCode, NextIndentStr, IndentStr]).

% Flex layout
generate_layout(flex, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(direction, Options, Dir, row),
    get_option(gap, Options, Gap, 0),
    get_option(justify, Options, Justify, start),
    get_option(align, Options, Align, stretch),
    get_option(wrap, Options, Wrap, false),

    (Dir = column -> Widget = 'Column' ; Widget = 'Row'),
    main_axis_alignment(Justify, MainAxis),
    cross_axis_alignment(Align, CrossAxis),

    (Wrap = true ->
        generate_children_with_spacing(Children, Gap, Indent, GenOpts, ChildrenCode),
        indent_string(Indent, IndentStr),
        NextIndent is Indent + 1,
        indent_string(NextIndent, NextIndentStr),
        format(atom(Code), '~wWrap(~n~wspacing: ~w,~n~wrunSpacing: ~w,~n~wchildren: [~n~w~w],~n~w),~n',
               [IndentStr, NextIndentStr, Gap, NextIndentStr, Gap, NextIndentStr, ChildrenCode, NextIndentStr, IndentStr])
    ;
        generate_children_with_spacing(Children, Gap, Indent, GenOpts, ChildrenCode),
        indent_string(Indent, IndentStr),
        NextIndent is Indent + 1,
        indent_string(NextIndent, NextIndentStr),
        format(atom(Code), '~w~w(~n~wmainAxisAlignment: ~w,~n~wcrossAxisAlignment: ~w,~n~wchildren: [~n~w~w],~n~w),~n',
               [IndentStr, Widget, NextIndentStr, MainAxis, NextIndentStr, CrossAxis,
                NextIndentStr, ChildrenCode, NextIndentStr, IndentStr])
    ).

% Grid layout
generate_layout(grid, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(columns, Options, Cols, 2),
    get_option(gap, Options, Gap, 0),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    NextIndent is Indent + 1,
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wGridView.count(~n~wcrossAxisCount: ~w,~n~wmainAxisSpacing: ~w,~n~wcrossAxisSpacing: ~w,~n~wshrinkWrap: true,~n~wchildren: [~n~w~w],~n~w),~n',
           [IndentStr, NextIndentStr, Cols, NextIndentStr, Gap, NextIndentStr, Gap,
            NextIndentStr, NextIndentStr, ChildrenCode, NextIndentStr, IndentStr]).

% Scroll layout
generate_layout(scroll, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(direction, Options, Dir, vertical),

    (Dir = vertical -> ScrollDir = 'Axis.vertical' ; ScrollDir = 'Axis.horizontal'),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    NextIndent is Indent + 1,
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wSingleChildScrollView(~n~wscrollDirection: ~w,~n~wchild: Column(~n~wchildren: [~n~w~w],~n~w),~n~w),~n',
           [IndentStr, NextIndentStr, ScrollDir, NextIndentStr, NextIndentStr, ChildrenCode,
            NextIndentStr, NextIndentStr, IndentStr]).

% Center layout
generate_layout(center, _Options, Children, Indent, GenOpts, Code) :- !,
    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    NextIndent is Indent + 1,
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wCenter(~n~wchild: Column(~n~wmainAxisSize: MainAxisSize.min,~n~wchildren: [~n~w~w],~n~w),~n~w),~n',
           [IndentStr, NextIndentStr, NextIndentStr, NextIndentStr, ChildrenCode,
            NextIndentStr, NextIndentStr, IndentStr]).

% Wrap layout
generate_layout(wrap, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(gap, Options, Gap, 0),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    NextIndent is Indent + 1,
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wWrap(~n~wspacing: ~w,~n~wrunSpacing: ~w,~n~wchildren: [~n~w~w],~n~w),~n',
           [IndentStr, NextIndentStr, Gap, NextIndentStr, Gap, NextIndentStr, ChildrenCode,
            NextIndentStr, IndentStr]).

% Default layout fallback
generate_layout(Type, _Options, Children, Indent, GenOpts, Code) :-
    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w// Layout: ~w~n~wColumn(children: [~w]),~n',
           [IndentStr, Type, IndentStr, ChildrenCode]).

% ============================================================================
% CONTAINER GENERATION
% ============================================================================

%! generate_container(+Type, +Options, +Content, +Indent, +GenOpts, -Code) is det

% Panel container -> Container with decoration
generate_container(panel, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(background, Options, Bg, '#16213e'),
    get_option(padding, Options, Padding, 20),
    get_option(rounded, Options, Rounded, 5),

    color_to_flutter(Bg, FlutterColor),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wContainer(~n~wpadding: EdgeInsets.all(~w),~n~wdecoration: BoxDecoration(~n~w  color: ~w,~n~w  borderRadius: BorderRadius.circular(~w),~n~w),~n~wchild: ~w~w),~n',
           [IndentStr, NextIndentStr, Padding, NextIndentStr, NextIndentStr, FlutterColor,
            NextIndentStr, Rounded, NextIndentStr, NextIndentStr, ContentCode, IndentStr]).

% Card container
generate_container(card, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(padding, Options, Padding, 16),
    get_option(elevation, Options, Elevation, 2),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wCard(~n~welevation: ~w,~n~wchild: Padding(~n~w  padding: EdgeInsets.all(~w),~n~w  child: ~w~w),~n~w),~n',
           [IndentStr, NextIndentStr, Elevation, NextIndentStr, NextIndentStr, Padding,
            NextIndentStr, NextIndentStr, ContentCode, NextIndentStr, IndentStr]).

% Scroll container
generate_container(scroll, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(max_height, Options, MaxHeight, 400),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wConstrainedBox(~n~wconstraints: BoxConstraints(maxHeight: ~w),~n~wchild: SingleChildScrollView(~n~w  child: ~w~w),~n~w),~n',
           [IndentStr, NextIndentStr, MaxHeight, NextIndentStr, NextIndentStr, ContentCode,
            NextIndentStr, IndentStr]).

% Section container
generate_container(section, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, ''),
    get_option(padding, Options, Padding, 0),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    (Title \= '' ->
        format(atom(Code), '~wPadding(~n~wpadding: EdgeInsets.all(~w),~n~wchild: Column(~n~w  crossAxisAlignment: CrossAxisAlignment.start,~n~w  children: [~n~w    Text("~w", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),~n~w    SizedBox(height: 8),~n~w    ~w~w  ],~n~w),~n~w),~n',
               [IndentStr, NextIndentStr, Padding, NextIndentStr, NextIndentStr,
                NextIndentStr, NextIndentStr, Title, NextIndentStr, NextIndentStr,
                ContentCode, NextIndentStr, NextIndentStr, IndentStr])
    ;
        format(atom(Code), '~wPadding(~n~wpadding: EdgeInsets.all(~w),~n~wchild: ~w~w),~n',
               [IndentStr, NextIndentStr, Padding, NextIndentStr, ContentCode, IndentStr])
    ).

% Modal container
generate_container(modal, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, ''),
    get_option(on_close, Options, OnClose, ''),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code), '~wDialog(~n~wchild: Column(~n~w  mainAxisSize: MainAxisSize.min,~n~w  children: [~n~w    Row(~n~w      mainAxisAlignment: MainAxisAlignment.spaceBetween,~n~w      children: [~n~w        Text("~w", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),~n~w        IconButton(icon: Icon(Icons.close), onPressed: ~w),~n~w      ],~n~w    ),~n~w    SizedBox(height: 16),~n~w    ~w~w  ],~n~w),~n~w),~n',
           [IndentStr, NextIndentStr, NextIndentStr, NextIndentStr, NextIndentStr,
            NextIndentStr, NextIndentStr, NextIndentStr, Title, NextIndentStr, OnClose,
            NextIndentStr, NextIndentStr, NextIndentStr, NextIndentStr, ContentCode,
            NextIndentStr, NextIndentStr, IndentStr]).

% Default container fallback
generate_container(Type, _Options, Content, Indent, GenOpts, Code) :-
    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w// Container: ~w~n~wContainer(child: ~w),~n',
           [IndentStr, Type, IndentStr, ContentCode]).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%! generate_component(+Type, +Options, +Indent, +GenOpts, -Code) is det

% Text component
generate_component(text, Options, Indent, GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    get_option(style, Options, Style, ''),
    get_option(size, Options, Size, 0),
    get_option(color, Options, Color, ''),

    indent_string(Indent, IndentStr),

    build_text_style(Style, Size, Color, StyleCode),

    % Handle different content types: binding, function call, or literal
    content_to_dart(Content, GenOpts, DartContentRaw, IsLiteral),

    % For dynamic values (map access), add .toString() for Text widget
    (IsLiteral = false, needs_tostring(DartContentRaw) ->
        format(atom(DartContent), '~w.toString()', [DartContentRaw])
    ;
        DartContent = DartContentRaw
    ),

    (IsLiteral = true ->
        (StyleCode \= '' ->
            format(atom(Code), '~wText("~w", style: ~w),~n', [IndentStr, DartContent, StyleCode])
        ;
            format(atom(Code), '~wText("~w"),~n', [IndentStr, DartContent])
        )
    ;
        (StyleCode \= '' ->
            format(atom(Code), '~wText(~w, style: ~w),~n', [IndentStr, DartContent, StyleCode])
        ;
            format(atom(Code), '~wText(~w),~n', [IndentStr, DartContent])
        )
    ).

%! needs_tostring(+DartExpr) is semidet
%  Check if expression is a dynamic map access that needs .toString()
needs_tostring(Expr) :-
    atom(Expr),
    sub_atom(Expr, _, _, _, '['),
    \+ sub_atom(Expr, _, _, _, '(').  % Not already a function call

%! content_to_dart(+Content, -DartContent, -IsLiteral) is det
%  Convert content to Dart expression, indicating if it's a string literal.
%  Backward compatible version without Options.
content_to_dart(Content, DartContent, IsLiteral) :-
    content_to_dart(Content, [], DartContent, IsLiteral).

%! content_to_dart(+Content, +Options, -DartContent, -IsLiteral) is det
%  Convert content to Dart expression with loop variable context.
content_to_dart(var(X), Opts, DartX, false) :- !, dotted_to_dart(X, Opts, DartX).
content_to_dart(bind(X), Opts, DartX, false) :- !, dotted_to_dart(X, Opts, DartX).
content_to_dart(format_size(Arg), Opts, DartCode, false) :- !,
    format_func_to_dart(format_size(Arg), Opts, DartCode).
content_to_dart(Content, Opts, DartContent, false) :-
    compound(Content),
    Content =.. [Func|Args],
    Func \= var, Func \= bind, !,
    snake_to_camel(Func, CamelFunc),
    maplist(arg_to_dart(Opts), Args, DartArgs),
    atomic_list_concat(DartArgs, ', ', ArgsStr),
    format(atom(DartContent), '~w(~w)', [CamelFunc, ArgsStr]).
content_to_dart(Content, _, Content, true) :- atom(Content), !.
content_to_dart(Content, _, Content, true).

% Heading component
generate_component(heading, Options, Indent, _GenOpts, Code) :- !,
    get_option(level, Options, Level, 1),
    get_option(content, Options, Content, ''),

    heading_size(Level, Size),
    indent_string(Indent, IndentStr),

    (is_binding(Content) ->
        binding_to_dart(Content, DartBinding),
        format(atom(Code), '~wText(~w, style: TextStyle(fontSize: ~w, fontWeight: FontWeight.bold)),~n',
               [IndentStr, DartBinding, Size])
    ;
        format(atom(Code), '~wText("~w", style: TextStyle(fontSize: ~w, fontWeight: FontWeight.bold)),~n',
               [IndentStr, Content, Size])
    ).

% Label component
generate_component(label, Options, Indent, _GenOpts, Code) :- !,
    get_option(text, Options, Text, ''),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~wText("~w", style: TextStyle(color: Colors.grey)),~n', [IndentStr, Text]).

% Button component
generate_component(button, Options, Indent, _GenOpts, Code) :- !,
    get_option_label(Options, Label),
    get_option(on_click, Options, OnClick, ''),
    get_option(variant, Options, Variant, primary),
    get_option(disabled, Options, Disabled, false),

    indent_string(Indent, IndentStr),
    button_widget(Variant, ButtonWidget),
    label_to_flutter(Label, LabelCode),
    onclick_to_dart(OnClick, DartOnClick),

    (Disabled \= false ->
        format(atom(Code), '~w~w(~n~w  onPressed: null,~n~w  child: ~w,~n~w),~n',
               [IndentStr, ButtonWidget, IndentStr, IndentStr, LabelCode, IndentStr])
    ;
        format(atom(Code), '~w~w(~n~w  onPressed: ~w,~n~w  child: ~w,~n~w),~n',
               [IndentStr, ButtonWidget, IndentStr, DartOnClick, IndentStr, LabelCode, IndentStr])
    ).

% Icon button component
generate_component(icon_button, Options, Indent, _GenOpts, Code) :- !,
    get_option(icon, Options, Icon, ''),
    get_option(on_click, Options, OnClick, ''),

    indent_string(Indent, IndentStr),
    icon_to_flutter(Icon, FlutterIcon),
    onclick_to_dart(OnClick, DartOnClick),

    format(atom(Code), '~wIconButton(~n~w  icon: Icon(~w),~n~w  onPressed: ~w,~n~w),~n',
           [IndentStr, IndentStr, FlutterIcon, IndentStr, DartOnClick, IndentStr]).

% Text input component
generate_component(text_input, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(placeholder, Options, Placeholder, ''),
    get_option(label, Options, Label, ''),
    get_option(type, Options, Type, text),

    indent_string(Indent, IndentStr),
    input_obscure(Type, Obscure),

    (Label \= '' ->
        format(atom(Code), '~wTextField(~n~w  controller: ~wController,~n~w  decoration: InputDecoration(~n~w    labelText: "~w",~n~w    hintText: "~w",~n~w    border: OutlineInputBorder(),~n~w  ),~n~w  obscureText: ~w,~n~w),~n',
               [IndentStr, IndentStr, Bind, IndentStr, IndentStr, Label, IndentStr, Placeholder,
                IndentStr, IndentStr, IndentStr, Obscure, IndentStr])
    ;
        format(atom(Code), '~wTextField(~n~w  controller: ~wController,~n~w  decoration: InputDecoration(~n~w    hintText: "~w",~n~w    border: OutlineInputBorder(),~n~w  ),~n~w  obscureText: ~w,~n~w),~n',
               [IndentStr, IndentStr, Bind, IndentStr, IndentStr, Placeholder,
                IndentStr, IndentStr, IndentStr, Obscure, IndentStr])
    ).

% Textarea component
generate_component(textarea, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(placeholder, Options, Placeholder, ''),
    get_option(rows, Options, Rows, 4),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),

    (Label \= '' ->
        format(atom(Code), '~wTextField(~n~w  controller: ~wController,~n~w  maxLines: ~w,~n~w  decoration: InputDecoration(~n~w    labelText: "~w",~n~w    hintText: "~w",~n~w    border: OutlineInputBorder(),~n~w  ),~n~w),~n',
               [IndentStr, IndentStr, Bind, IndentStr, Rows, IndentStr, IndentStr, Label,
                IndentStr, Placeholder, IndentStr, IndentStr, IndentStr])
    ;
        format(atom(Code), '~wTextField(~n~w  controller: ~wController,~n~w  maxLines: ~w,~n~w  decoration: InputDecoration(~n~w    hintText: "~w",~n~w    border: OutlineInputBorder(),~n~w  ),~n~w),~n',
               [IndentStr, IndentStr, Bind, IndentStr, Rows, IndentStr, IndentStr, Placeholder,
                IndentStr, IndentStr, IndentStr])
    ).

% Checkbox component
generate_component(checkbox, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~wRow(~n~w  children: [~n~w    Checkbox(~n~w      value: ~w,~n~w      onChanged: (value) => setState(() => ~w = value!),~n~w    ),~n~w    Text("~w"),~n~w  ],~n~w),~n',
           [IndentStr, IndentStr, IndentStr, IndentStr, Bind, IndentStr, Bind, IndentStr,
            IndentStr, Label, IndentStr, IndentStr]).

% Select component (DropdownButton)
generate_component(select, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(options, Options, Opts, []),
    get_option(placeholder, Options, Placeholder, 'Select...'),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),

    (is_binding(Opts) ->
        binding_to_dart(Opts, DartOpts),
        format(atom(ItemsCode), '~w.map((opt) => DropdownMenuItem(value: opt, child: Text(opt))).toList()', [DartOpts])
    ;
        generate_dropdown_items(Opts, ItemsCode)
    ),

    (Label \= '' ->
        format(atom(Code), '~wColumn(~n~w  crossAxisAlignment: CrossAxisAlignment.start,~n~w  children: [~n~w    Text("~w", style: TextStyle(color: Colors.grey)),~n~w    DropdownButton(~n~w      value: ~w,~n~w      hint: Text("~w"),~n~w      items: ~w,~n~w      onChanged: (value) => setState(() => ~w = value),~n~w    ),~n~w  ],~n~w),~n',
               [IndentStr, IndentStr, IndentStr, IndentStr, Label, IndentStr, IndentStr, Bind,
                IndentStr, Placeholder, IndentStr, ItemsCode, IndentStr, Bind, IndentStr,
                IndentStr, IndentStr])
    ;
        format(atom(Code), '~wDropdownButton(~n~w  value: ~w,~n~w  hint: Text("~w"),~n~w  items: ~w,~n~w  onChanged: (value) => setState(() => ~w = value),~n~w),~n',
               [IndentStr, IndentStr, Bind, IndentStr, Placeholder, IndentStr, ItemsCode,
                IndentStr, Bind, IndentStr])
    ).

% Switch component
generate_component(switch, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~wRow(~n~w  children: [~n~w    Switch(~n~w      value: ~w,~n~w      onChanged: (value) => setState(() => ~w = value),~n~w    ),~n~w    Text("~w"),~n~w  ],~n~w),~n',
           [IndentStr, IndentStr, IndentStr, IndentStr, Bind, IndentStr, Bind, IndentStr,
            IndentStr, Label, IndentStr, IndentStr]).

% Spinner component
generate_component(spinner, Options, Indent, _GenOpts, Code) :- !,
    get_option(size, Options, Size, 24),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~wSizedBox(~n~w  width: ~w,~n~w  height: ~w,~n~w  child: CircularProgressIndicator(),~n~w),~n',
           [IndentStr, IndentStr, Size, IndentStr, Size, IndentStr, IndentStr]).

% Progress component
generate_component(progress, Options, Indent, _GenOpts, Code) :- !,
    get_option(value, Options, Value, 0),
    get_option(max, Options, Max, 100),

    indent_string(Indent, IndentStr),
    (is_binding(Value) ->
        binding_to_dart(Value, DartValue),
        format(atom(Code), '~wLinearProgressIndicator(value: ~w / ~w),~n', [IndentStr, DartValue, Max])
    ;
        ProgressValue is Value / Max,
        format(atom(Code), '~wLinearProgressIndicator(value: ~w),~n', [IndentStr, ProgressValue])
    ).

% Badge component
generate_component(badge, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    get_option(variant, Options, Variant, info),

    variant_badge_color(Variant, Color),
    indent_string(Indent, IndentStr),

    (is_binding(Content) ->
        binding_to_dart(Content, DartContent),
        format(atom(Code), '~wContainer(~n~w  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 2),~n~w  decoration: BoxDecoration(~n~w    color: ~w,~n~w    borderRadius: BorderRadius.circular(3),~n~w  ),~n~w  child: Text(~w, style: TextStyle(color: Colors.white, fontSize: 12)),~n~w),~n',
               [IndentStr, IndentStr, IndentStr, IndentStr, Color, IndentStr, IndentStr,
                IndentStr, DartContent, IndentStr])
    ;
        format(atom(Code), '~wContainer(~n~w  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 2),~n~w  decoration: BoxDecoration(~n~w    color: ~w,~n~w    borderRadius: BorderRadius.circular(3),~n~w  ),~n~w  child: Text("~w", style: TextStyle(color: Colors.white, fontSize: 12)),~n~w),~n',
               [IndentStr, IndentStr, IndentStr, IndentStr, Color, IndentStr, IndentStr,
                IndentStr, Content, IndentStr])
    ).

% Divider component
generate_component(divider, Options, Indent, _GenOpts, Code) :- !,
    get_option(margin, Options, Margin, 16),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~wPadding(~n~w  padding: EdgeInsets.symmetric(vertical: ~w),~n~w  child: Divider(),~n~w),~n',
           [IndentStr, IndentStr, Margin, IndentStr, IndentStr]).

% Spacer component
generate_component(spacer, Options, Indent, _GenOpts, Code) :- !,
    get_option(size, Options, Size, 16),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~wSizedBox(height: ~w),~n', [IndentStr, Size]).

% Image component
generate_component(image, Options, Indent, _GenOpts, Code) :- !,
    get_option(src, Options, Src, ''),
    get_option(width, Options, Width, ''),
    get_option(height, Options, Height, ''),

    indent_string(Indent, IndentStr),

    build_image_dimensions(Width, Height, DimCode),

    (is_binding(Src) ->
        binding_to_dart(Src, DartSrc),
        format(atom(Code), '~wImage.network(~w~w),~n', [IndentStr, DartSrc, DimCode])
    ;
        format(atom(Code), '~wImage.network("~w"~w),~n', [IndentStr, Src, DimCode])
    ).

% Avatar component
generate_component(avatar, Options, Indent, _GenOpts, Code) :- !,
    get_option(src, Options, Src, ''),
    get_option(name, Options, Name, ''),
    get_option(size, Options, Size, 40),

    Radius is Size / 2,
    indent_string(Indent, IndentStr),

    (Src \= '' ->
        format(atom(Code), '~wCircleAvatar(~n~w  radius: ~w,~n~w  backgroundImage: NetworkImage("~w"),~n~w),~n',
               [IndentStr, IndentStr, Radius, IndentStr, Src, IndentStr])
    ;
        format(atom(Code), '~wCircleAvatar(~n~w  radius: ~w,~n~w  child: Text("~w"),~n~w),~n',
               [IndentStr, IndentStr, Radius, IndentStr, Name, IndentStr])
    ).

% Icon component
generate_component(icon, Options, Indent, _GenOpts, Code) :- !,
    get_option(name, Options, Name, ''),
    get_option(size, Options, Size, 24),

    indent_string(Indent, IndentStr),
    icon_to_flutter(Name, FlutterIcon),
    format(atom(Code), '~wIcon(~w, size: ~w),~n', [IndentStr, FlutterIcon, Size]).

% Code component
generate_component(code, Options, Indent, GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),

    indent_string(Indent, IndentStr),
    (is_binding(Content) ->
        binding_to_dart(Content, GenOpts, DartContentRaw),
        % Add .toString() for dynamic map access
        (needs_tostring(DartContentRaw) ->
            format(atom(DartContent), '~w.toString()', [DartContentRaw])
        ;
            DartContent = DartContentRaw
        ),
        format(atom(Code), '~wContainer(~n~w  padding: EdgeInsets.all(8),~n~w  decoration: BoxDecoration(~n~w    color: Color(0xFF1a1a2e),~n~w    borderRadius: BorderRadius.circular(3),~n~w  ),~n~w  child: Text(~w, style: TextStyle(fontFamily: "monospace")),~n~w),~n',
               [IndentStr, IndentStr, IndentStr, IndentStr, IndentStr, IndentStr, IndentStr, DartContent, IndentStr])
    ;
        format(atom(Code), '~wContainer(~n~w  padding: EdgeInsets.all(8),~n~w  decoration: BoxDecoration(~n~w    color: Color(0xFF1a1a2e),~n~w    borderRadius: BorderRadius.circular(3),~n~w  ),~n~w  child: Text("~w", style: TextStyle(fontFamily: "monospace")),~n~w),~n',
               [IndentStr, IndentStr, IndentStr, IndentStr, IndentStr, IndentStr, IndentStr, Content, IndentStr])
    ).

% Pre component
generate_component(pre, Options, Indent, GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),

    indent_string(Indent, IndentStr),
    (is_binding(Content) ->
        binding_to_dart(Content, GenOpts, DartContentRaw),
        % Add .toString() for dynamic map access
        (needs_tostring(DartContentRaw) ->
            format(atom(DartContent), '~w.toString()', [DartContentRaw])
        ;
            DartContent = DartContentRaw
        ),
        format(atom(Code), '~wText(~w, style: TextStyle(fontFamily: "monospace")),~n', [IndentStr, DartContent])
    ;
        format(atom(Code), '~wText("~w", style: TextStyle(fontFamily: "monospace")),~n', [IndentStr, Content])
    ).

% Alert component
generate_component(alert, Options, Indent, _GenOpts, Code) :- !,
    get_option(message, Options, Message, ''),
    get_option(variant, Options, Variant, info),

    variant_alert_colors(Variant, BgColor, TextColor),
    indent_string(Indent, IndentStr),

    format(atom(Code), '~wContainer(~n~w  padding: EdgeInsets.all(12),~n~w  decoration: BoxDecoration(~n~w    color: ~w,~n~w    borderRadius: BorderRadius.circular(5),~n~w  ),~n~w  child: Text("~w", style: TextStyle(color: ~w)),~n~w),~n',
           [IndentStr, IndentStr, IndentStr, IndentStr, BgColor, IndentStr, IndentStr,
            IndentStr, Message, TextColor, IndentStr]).

% Link component
generate_component(link, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, ''),
    get_option(on_click, Options, OnClick, ''),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~wGestureDetector(~n~w  onTap: ~w,~n~w  child: Text("~w", style: TextStyle(color: Colors.blue, decoration: TextDecoration.underline)),~n~w),~n',
           [IndentStr, IndentStr, OnClick, IndentStr, Label, IndentStr]).

% Tabs component
generate_component(tabs, Options, Indent, _GenOpts, Code) :- !,
    get_option(items, Options, Items, []),
    get_option(active, Options, _Active, ''),  % Active index tracked by TabController
    get_option(on_change, Options, OnChange, ''),

    indent_string(Indent, IndentStr),

    (is_binding(Items) ->
        binding_to_dart(Items, DartItems),
        format(atom(TabsCode), '~w.map((item) => Tab(text: item)).toList()', [DartItems])
    ;
        generate_tab_items(Items, TabsCode)
    ),

    format(atom(Code), '~wTabBar(~n~w  tabs: ~w,~n~w  onTap: ~w,~n~w),~n',
           [IndentStr, IndentStr, TabsCode, IndentStr, OnChange, IndentStr]).

% Default component fallback
generate_component(Type, Options, Indent, _GenOpts, Code) :-
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w// Component: ~w ~w~n', [IndentStr, Type, Options]).

% ============================================================================
% CONDITIONAL GENERATION
% ============================================================================

generate_conditional(Condition, Content, Indent, GenOpts, Code) :- !,
    condition_to_dart(Condition, GenOpts, DartCondRaw),
    % In Dart, dynamic values need explicit null check for truthiness
    make_dart_truthy(DartCondRaw, DartCond),
    generate_node(Content, Indent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~wif (~w) ~w', [IndentStr, DartCond, ContentCode]).

%! make_dart_truthy(+RawCond, -DartCond) is det
%  Wrap simple variable/property access with != null for Dart truthiness.
%  Don't wrap if it already contains comparison operators.
make_dart_truthy(Cond, Result) :-
    atom(Cond),
    atom_string(Cond, CondStr),
    % Don't add != null if already a boolean expression
    (   (sub_string(CondStr, _, _, _, "==");
         sub_string(CondStr, _, _, _, "!=");
         sub_string(CondStr, _, _, _, "&&");
         sub_string(CondStr, _, _, _, "||");
         sub_string(CondStr, _, _, _, ".isEmpty");
         sub_string(CondStr, _, _, _, ".isNotEmpty"))
    ->  Result = Cond
    ;   % Check if it's a map access or variable that needs null check
        (   (sub_atom(Cond, _, _, _, '['); sub_atom(Cond, 0, 1, _, '_'))
        ->  format(atom(Result), '~w != null', [Cond])
        ;   Result = Cond
        )
    ).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

%! generate_children(+Children, +Indent, +Options, -Code) is det
generate_children([], _Indent, _Options, '') :- !.
generate_children([Child|Rest], Indent, Options, Code) :-
    NextIndent is Indent + 2,
    generate_node(Child, NextIndent, Options, ChildCode),
    generate_children(Rest, Indent, Options, RestCode),
    atom_concat(ChildCode, RestCode, Code).

%! generate_children_with_spacing(+Children, +Spacing, +Indent, +Options, -Code) is det
generate_children_with_spacing([], _Spacing, _Indent, _Options, '') :- !.
generate_children_with_spacing([Child], _Spacing, Indent, Options, Code) :- !,
    NextIndent is Indent + 2,
    generate_node(Child, NextIndent, Options, Code).
generate_children_with_spacing([Child|Rest], Spacing, Indent, Options, Code) :-
    NextIndent is Indent + 2,
    generate_node(Child, NextIndent, Options, ChildCode),
    indent_string(NextIndent, NextIndentStr),
    format(atom(SpacerCode), '~wSizedBox(height: ~w),~n', [NextIndentStr, Spacing]),
    generate_children_with_spacing(Rest, Spacing, Indent, Options, RestCode),
    atomic_list_concat([ChildCode, SpacerCode, RestCode], Code).

%! indent_string(+Level, -String) is det
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
%  Extracts label from options, handling both simple label(X) and conditional label(Cond, TrueVal, FalseVal)
get_option_label(Options, conditional(Cond, TrueVal, FalseVal)) :-
    member(label(Cond, TrueVal, FalseVal), Options), !.
get_option_label(Options, simple(Label)) :-
    member(label(Label), Options), !.
get_option_label(_, simple('Button')).

%! label_to_flutter(+Label, -Code) is det
%  Converts label spec to Flutter Text widget code
label_to_flutter(simple(Text), Code) :- !,
    format(atom(Code), 'Text("~w")', [Text]).
label_to_flutter(conditional(Cond, TrueVal, FalseVal), Code) :- !,
    condition_to_dart(Cond, DartCond),
    format(atom(Code), 'Text(~w ? "~w" : "~w")', [DartCond, TrueVal, FalseVal]).
label_to_flutter(Text, Code) :-
    format(atom(Code), 'Text("~w")', [Text]).

% Alignment helpers
main_axis_alignment(start, 'MainAxisAlignment.start').
main_axis_alignment(center, 'MainAxisAlignment.center').
main_axis_alignment(end, 'MainAxisAlignment.end').
main_axis_alignment(between, 'MainAxisAlignment.spaceBetween').
main_axis_alignment(around, 'MainAxisAlignment.spaceAround').
main_axis_alignment(evenly, 'MainAxisAlignment.spaceEvenly').
main_axis_alignment(_, 'MainAxisAlignment.start').

cross_axis_alignment(start, 'CrossAxisAlignment.start').
cross_axis_alignment(center, 'CrossAxisAlignment.center').
cross_axis_alignment(end, 'CrossAxisAlignment.end').
cross_axis_alignment(stretch, 'CrossAxisAlignment.stretch').
cross_axis_alignment(_, 'CrossAxisAlignment.start').

% Heading sizes
heading_size(1, 32).
heading_size(2, 24).
heading_size(3, 20).
heading_size(4, 18).
heading_size(5, 16).
heading_size(6, 14).
heading_size(_, 16).

% Button widgets
button_widget(primary, 'ElevatedButton').
button_widget(secondary, 'OutlinedButton').
button_widget(danger, 'ElevatedButton').
button_widget(ghost, 'TextButton').
button_widget(_, 'ElevatedButton').

% Input obscure
input_obscure(password, 'true').
input_obscure(_, 'false').

% Color conversion
color_to_flutter(Color, FlutterColor) :-
    atom(Color),
    atom_codes(Color, [0'#|HexCodes]),
    atom_codes(Hex, HexCodes),
    format(atom(FlutterColor), 'Color(0xFF~w)', [Hex]).
color_to_flutter(Color, Color) :- \+ atom(Color).

% Icon conversion (basic emoji to Icons mapping)
icon_to_flutter(Name, FlutterIcon) :-
    (Name = folder ; Name = '📁') -> FlutterIcon = 'Icons.folder' ;
    (Name = file ; Name = '📄') -> FlutterIcon = 'Icons.insert_drive_file' ;
    (Name = search ; Name = '🔍') -> FlutterIcon = 'Icons.search' ;
    (Name = close ; Name = '❌') -> FlutterIcon = 'Icons.close' ;
    (Name = menu ; Name = '☰') -> FlutterIcon = 'Icons.menu' ;
    FlutterIcon = 'Icons.circle'.

% Variant colors
variant_badge_color(info, 'Colors.blue').
variant_badge_color(success, 'Colors.green').
variant_badge_color(warning, 'Colors.orange').
variant_badge_color(error, 'Colors.red').
variant_badge_color(_, 'Colors.grey').

variant_alert_colors(info, 'Color(0xFFd1ecf1)', 'Color(0xFF0c5460)').
variant_alert_colors(success, 'Color(0xFFd4edda)', 'Color(0xFF155724)').
variant_alert_colors(warning, 'Color(0xFFfff3cd)', 'Color(0xFF856404)').
variant_alert_colors(error, 'Color(0xFFf8d7da)', 'Color(0xFF721c24)').
variant_alert_colors(_, 'Colors.grey.shade200', 'Colors.black').

% Text style builder
build_text_style('', 0, '', '') :- !.
build_text_style(Style, Size, Color, StyleCode) :-
    findall(S, (
        (Style = muted -> S = 'color: Colors.grey' ;
         Style = error -> S = 'color: Colors.red' ;
         Style = success -> S = 'color: Colors.green' ;
         fail),
        S \= ''
    ), Styles1),
    (Size > 0 -> format(atom(SizeS), 'fontSize: ~w', [Size]), Styles2 = [SizeS|Styles1] ; Styles2 = Styles1),
    (Color \= '' -> color_to_flutter(Color, FC), format(atom(ColorS), 'color: ~w', [FC]), Styles3 = [ColorS|Styles2] ; Styles3 = Styles2),
    (Styles3 \= [] ->
        atomic_list_concat(Styles3, ', ', StyleStr),
        format(atom(StyleCode), 'TextStyle(~w)', [StyleStr])
    ;
        StyleCode = ''
    ).

% Image dimensions builder
build_image_dimensions('', '', '') :- !.
build_image_dimensions(W, '', Code) :- W \= '', !, format(atom(Code), ', width: ~w', [W]).
build_image_dimensions('', H, Code) :- H \= '', !, format(atom(Code), ', height: ~w', [H]).
build_image_dimensions(W, H, Code) :- format(atom(Code), ', width: ~w, height: ~w', [W, H]).

% Binding detection
is_binding(var(_)) :- !.
is_binding(bind(_)) :- !.
is_binding(Term) :- atom(Term), atom_codes(Term, [C|_]), C \= 34, C \= 39.

% Binding conversion - with backward compatibility
binding_to_dart(X, DartX) :- binding_to_dart(X, [], DartX).

binding_to_dart(var(X), Opts, DartX) :- !, dotted_to_dart(X, Opts, DartX).
binding_to_dart(bind(X), Opts, DartX) :- !, dotted_to_dart(X, Opts, DartX).
binding_to_dart(X, Opts, DartX) :- atom(X), !, dotted_to_dart(X, Opts, DartX).
binding_to_dart(X, _, X).

% Condition conversion for Dart - with backward compatibility
condition_to_dart(Cond, DartCond) :- condition_to_dart(Cond, [], DartCond).

condition_to_dart(var(X), Opts, DartX) :- !,
    dotted_to_dart(X, Opts, DartX).
condition_to_dart(Cond, Opts, DartCond) :- atom(Cond), !,
    dotted_to_dart(Cond, Opts, DartCond).
condition_to_dart(not(C), Opts, DartCond) :- !,
    condition_to_dart(C, Opts, VC),
    format(atom(DartCond), '!~w', [VC]).
condition_to_dart(and(A, B), Opts, DartCond) :- !,
    condition_to_dart(A, Opts, VA),
    condition_to_dart(B, Opts, VB),
    format(atom(DartCond), '(~w && ~w)', [VA, VB]).
condition_to_dart(or(A, B), Opts, DartCond) :- !,
    condition_to_dart(A, Opts, VA),
    condition_to_dart(B, Opts, VB),
    format(atom(DartCond), '(~w || ~w)', [VA, VB]).
condition_to_dart(eq(A, B), Opts, DartCond) :- !,
    condition_to_dart(A, Opts, VA),
    condition_to_dart(B, Opts, VB),
    format(atom(DartCond), '(~w == ~w)', [VA, VB]).
condition_to_dart(not_eq(A, B), Opts, DartCond) :- !,
    condition_to_dart(A, Opts, VA),
    condition_to_dart(B, Opts, VB),
    format(atom(DartCond), '(~w != ~w)', [VA, VB]).
condition_to_dart(empty(C), Opts, DartCond) :- !,
    condition_to_dart(C, Opts, VC),
    format(atom(DartCond), '(~w == null || ~w.isEmpty)', [VC, VC]).
condition_to_dart([H|T], Opts, DartCond) :- !,
    condition_to_dart(H, Opts, VH),
    condition_to_dart(T, Opts, VT),
    format(atom(DartCond), '(~w && ~w)', [VH, VT]).
condition_to_dart([], _, 'true') :- !.
condition_to_dart(Term, _, Dart) :-
    term_to_atom(Term, Dart).

items_to_dart(var(X), DartX) :- !, dotted_to_dart(X, DartX).
items_to_dart(Items, Items) :- atom(Items), !.
items_to_dart(Items, DartItems) :- is_list(Items), !, term_to_atom(Items, DartItems).
items_to_dart(Term, Dart) :- term_to_atom(Term, Dart).

% ============================================================================
% DART NAMING CONVENTIONS
% ============================================================================

%! snake_to_camel(+SnakeName, -CamelName) is det
%  Convert snake_case to camelCase for Dart naming conventions.
snake_to_camel(Name, CamelName) :-
    atom_string(Name, NameStr),
    split_string(NameStr, "_", "", Parts),
    Parts = [First|Rest],
    maplist(capitalize_first, Rest, CapRest),
    append([First], CapRest, AllParts),
    atomics_to_string(AllParts, CamelStr),
    atom_string(CamelName, CamelStr).

capitalize_first(Str, Cap) :-
    string_chars(Str, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HUC]),
    string_chars(Cap, [HUC|T]).
capitalize_first("", "").

%! dotted_to_dart(+Name, -DartName) is det
%  Convert dotted names like browse.path to Dart map access _browse['path']
%  or to proper camelCase state variable access.
%  For loop variables (entry, item, etc), no underscore prefix is added.
dotted_to_dart(Name, DartName) :-
    dotted_to_dart(Name, [], DartName).

%! dotted_to_dart(+Name, +Options, -DartName) is det
%  Convert with loop variable context.
%  State variables get underscore prefix, loop variables don't.
dotted_to_dart(Name, Options, DartName) :-
    atom_string(Name, NameStr),
    get_option(loop_vars, Options, LoopVars, []),
    (   sub_string(NameStr, _, _, _, ".")
    ->  split_string(NameStr, ".", "", Parts),
        Parts = [Obj|Fields],
        atom_string(ObjAtom, Obj),
        (   member(ObjAtom, LoopVars)
        ->  % Loop variable - no underscore prefix
            foldl(add_field_access, Fields, Obj, DartName)
        ;   % State variable - add underscore prefix
            snake_to_camel_str(Obj, CamelObj),
            format(string(DartStr), "_~w", [CamelObj]),
            foldl(add_field_access, Fields, DartStr, DartName)
        )
    ;   atom_string(NameAtom, NameStr),
        (   member(NameAtom, LoopVars)
        ->  DartName = Name
        ;   % State variable - add underscore prefix for Dart convention
            snake_to_camel(Name, CamelName),
            format(atom(DartName), '_~w', [CamelName])
        )
    ).

snake_to_camel_str(Str, CamelStr) :-
    atom_string(Atom, Str),
    snake_to_camel(Atom, CamelAtom),
    atom_string(CamelAtom, CamelStr).

add_field_access(Field, Acc, Result) :-
    format(atom(Result), "~w['~w']", [Acc, Field]).

%! onclick_to_dart(+OnClick, -DartCode) is det
%  Convert onclick handler to Dart function call syntax.
onclick_to_dart('', 'null') :- !.
onclick_to_dart(null, 'null') :- !.
onclick_to_dart(navigate_up, '() => navigateUp()') :- !.
onclick_to_dart(view_file, '() => viewFile()') :- !.
onclick_to_dart(download_file, '() => downloadFile()') :- !.
onclick_to_dart(search_here, '() => searchHere()') :- !.
onclick_to_dart(set_working_dir, '() => setWorkingDir()') :- !.
onclick_to_dart(handle_entry_click(var(Entry)), DartCode) :- !,
    dotted_to_dart(Entry, DartEntry),
    format(atom(DartCode), '() => handleEntryClick(~w)', [DartEntry]).
onclick_to_dart(Term, DartCode) :-
    compound(Term), !,
    Term =.. [Name|Args],
    snake_to_camel(Name, CamelName),
    (   Args = []
    ->  format(atom(DartCode), '() => ~w()', [CamelName])
    ;   maplist(arg_to_dart, Args, DartArgs),
        atomic_list_concat(DartArgs, ', ', ArgsStr),
        format(atom(DartCode), '() => ~w(~w)', [CamelName, ArgsStr])
    ).
onclick_to_dart(Name, DartCode) :-
    atom(Name), !,
    snake_to_camel(Name, CamelName),
    format(atom(DartCode), '() => ~w()', [CamelName]).

% Backward compatible version
arg_to_dart(X, Dart) :- arg_to_dart([], X, Dart).

arg_to_dart(Opts, var(X), Dart) :- !, dotted_to_dart(X, Opts, Dart).
arg_to_dart(Opts, bind(X), Dart) :- !, dotted_to_dart(X, Opts, Dart).
arg_to_dart(_, X, X) :- atom(X), !.
arg_to_dart(_, X, Dart) :- term_to_atom(X, Dart).

%! format_func_to_dart(+FuncCall, -DartCode) is det
%  Convert format functions like format_size(var(X)) to formatSize(x).
%  Backward compatible version.
format_func_to_dart(FuncCall, DartCode) :- format_func_to_dart(FuncCall, [], DartCode).

format_func_to_dart(format_size(var(X)), Opts, DartCode) :- !,
    dotted_to_dart(X, Opts, DartX),
    format(atom(DartCode), 'formatSize(~w)', [DartX]).
format_func_to_dart(format_size(bind(X)), Opts, DartCode) :- !,
    dotted_to_dart(X, Opts, DartX),
    format(atom(DartCode), 'formatSize(~w)', [DartX]).
format_func_to_dart(format_size(X), _, DartCode) :- !,
    format(atom(DartCode), 'formatSize(~w)', [X]).
format_func_to_dart(X, _, X).

% Dropdown items generator
generate_dropdown_items([], '[]') :- !.
generate_dropdown_items(Items, Code) :-
    findall(ItemCode, (
        member(Item, Items),
        (Item = opt(Value, Label) ->
            format(atom(ItemCode), 'DropdownMenuItem(value: "~w", child: Text("~w"))', [Value, Label])
        ;
            format(atom(ItemCode), 'DropdownMenuItem(value: "~w", child: Text("~w"))', [Item, Item])
        )
    ), ItemCodes),
    atomic_list_concat(ItemCodes, ', ', ItemsStr),
    format(atom(Code), '[~w]', [ItemsStr]).

% Tab items generator
generate_tab_items([], '[]') :- !.
generate_tab_items(Items, Code) :-
    findall(TabCode, (
        member(Item, Items),
        format(atom(TabCode), 'Tab(text: "~w")', [Item])
    ), TabCodes),
    atomic_list_concat(TabCodes, ', ', TabsStr),
    format(atom(Code), '[~w]', [TabsStr]).

% ============================================================================
% FULL FILE GENERATION
% ============================================================================

%! generate_flutter_file(+UISpec, +Options, -File) is det
%  Generate a complete Flutter widget file.
generate_flutter_file(UISpec, Options, File) :-
    get_option(name, Options, Name, 'GeneratedWidget'),
    generate_flutter_widget(UISpec, Options, Widget),
    format(atom(File),
'import \'package:flutter/material.dart\';

class ~w extends StatefulWidget {
  const ~w({super.key});

  @override
  State<~w> createState() => _~wState();
}

class _~wState extends State<~w> {
  // TODO: Add state variables

  @override
  Widget build(BuildContext context) {
    return ~w  }
}
', [Name, Name, Name, Name, Name, Name, Widget]).

% ============================================================================
% TESTING
% ============================================================================

test_flutter_generator :-
    format('~n=== Flutter Generator Tests ===~n~n'),

    % Test 1: Simple text
    format('Test 1: Simple text component...~n'),
    generate_flutter_widget(component(text, [content("Hello World")]), [], T1),
    format('  Output: ~w~n', [T1]),

    % Test 2: Button
    format('~nTest 2: Button component...~n'),
    generate_flutter_widget(component(button, [label("Click Me"), on_click(handleClick)]), [], T2),
    format('  Output: ~w~n', [T2]),

    % Test 3: Stack layout
    format('~nTest 3: Stack layout (Column)...~n'),
    UI3 = layout(stack, [spacing(16)], [
        component(heading, [level(1), content("Title")]),
        component(text, [content("Description")])
    ]),
    generate_flutter_widget(UI3, [], T3),
    format('  Output:~n~w~n', [T3]),

    % Test 4: Form with inputs
    format('~nTest 4: Form with inputs...~n'),
    UI4 = layout(stack, [spacing(12)], [
        component(text_input, [label("Email"), bind(email), placeholder("Enter email")]),
        component(text_input, [label("Password"), type(password), bind(password)]),
        component(button, [label("Submit"), on_click(submit), variant(primary)])
    ]),
    generate_flutter_widget(UI4, [], T4),
    format('  Output:~n~w~n', [T4]),

    % Test 5: Conditional
    format('~nTest 5: Conditional rendering...~n'),
    UI5 = when(isLoggedIn, component(text, [content("Welcome!")])),
    generate_flutter_widget(UI5, [], T5),
    format('  Output: ~w~n', [T5]),

    % Test 6: Foreach
    format('~nTest 6: Foreach iteration...~n'),
    UI6 = foreach(items, item, component(text, [content(item)])),
    generate_flutter_widget(UI6, [], T6),
    format('  Output: ~w~n', [T6]),

    % Test 7: Nested containers
    format('~nTest 7: Nested containers...~n'),
    UI7 = container(panel, [padding(20), background('#1a1a2e')], [
        layout(stack, [spacing(8)], [
            component(heading, [level(2), content("Panel Title")]),
            component(text, [content("Panel content goes here")])
        ])
    ]),
    generate_flutter_widget(UI7, [], T7),
    format('  Output:~n~w~n', [T7]),

    % Test 8: Full file generation
    format('~nTest 8: Full file generation...~n'),
    UI8 = layout(stack, [spacing(16)], [
        component(heading, [level(1), content("My App")]),
        component(button, [label("Click"), on_click(handleClick)])
    ]),
    generate_flutter_file(UI8, [name('MyWidget')], F8),
    format('  Output:~n~w~n', [F8]),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('Flutter Generator module loaded~n')
), now).
