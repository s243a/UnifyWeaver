% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% react_generator.pl - Generate React/JSX code from UI primitives
%
% Compiles declarative UI specifications to React functional components
% using JSX syntax and hooks.
%
% Usage:
%   use_module('src/unifyweaver/ui/react_generator').
%   UI = layout(stack, [spacing(16)], [
%       component(text, [content("Hello")]),
%       component(button, [label("Click"), on_click(submit)])
%   ]),
%   generate_react_template(UI, Template).

:- module(react_generator, [
    % Template generation
    generate_react_template/2,      % generate_react_template(+UISpec, -Template)
    generate_react_template/3,      % generate_react_template(+UISpec, +Options, -Template)

    % Full component generation
    generate_react_component/3,     % generate_react_component(+UISpec, +Options, -Component)

    % Testing
    test_react_generator/0
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

% Load pattern system (optional - patterns expanded on-the-fly if available)
:- catch(use_module('ui_patterns'), _, true).

% ============================================================================
% TEMPLATE GENERATION
% ============================================================================

%! generate_react_template(+UISpec, -Template) is det
%  Generate React JSX template from UI specification.
generate_react_template(UISpec, Template) :-
    generate_react_template(UISpec, [], Template).

%! generate_react_template(+UISpec, +Options, -Template) is det
%  Generate React JSX template with options.
%  Options:
%    indent(N) - Starting indentation level
%    style(tailwind|css|inline) - Styling approach
generate_react_template(UISpec, Options, Template) :-
    get_option(indent, Options, Indent, 0),
    generate_node(UISpec, Indent, Options, Template).

% ============================================================================
% NODE GENERATION
% ============================================================================

%! generate_node(+Node, +Indent, +Options, -Code) is det
%  Generate React JSX code for any UI node.

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
        format(atom(Code), '~w{/* Pattern ~w not expanded: ui_patterns module not loaded */}~n', [IndentStr, Name])
    ), !.

% Conditional: when
generate_node(when(Condition, Content), Indent, Options, Code) :- !,
    generate_conditional(Condition, Content, Indent, Options, Code).

% Conditional: unless
generate_node(unless(Condition, Content), Indent, Options, Code) :- !,
    condition_to_jsx(Condition, JsxCond),
    generate_node(Content, Indent, Options, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w{!(~w) && (~n~w~w)}~n',
           [IndentStr, JsxCond, ContentCode, IndentStr]).

% Foreach iteration
generate_node(foreach(Items, Var, Template), Indent, Options, Code) :- !,
    items_to_jsx(Items, JsxItems),
    NextIndent is Indent + 1,
    generate_node(Template, NextIndent, Options, TemplateCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),
    format(atom(Code), '~w{~w.map((~w, index) => (~n~w<React.Fragment key={index}>~n~w~w</React.Fragment>~n~w))}~n',
           [IndentStr, JsxItems, Var, NextIndentStr, TemplateCode, NextIndentStr, IndentStr]).

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
    format(atom(Code), '~w{/* Unknown node: ~w */}~n', [IndentStr, Functor]).

% ============================================================================
% LAYOUT GENERATION
% ============================================================================

%! generate_layout(+Type, +Options, +Children, +Indent, +GenOpts, -Code) is det

% Stack layout -> div with flex-direction
generate_layout(stack, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(direction, Options, Dir, column),
    get_option(spacing, Options, Spacing, 0),
    get_option(align, Options, Align, stretch),
    get_option(padding, Options, Padding, 0),

    direction_style(Dir, DirStyle),
    spacing_style(Spacing, SpacingStyle),
    align_style(Align, AlignStyle),
    padding_style(Padding, PaddingStyle),

    build_style_object([DirStyle, SpacingStyle, AlignStyle, PaddingStyle], StyleObj),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ display: "flex", ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ChildrenCode, IndentStr]).

% Flex layout
generate_layout(flex, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(direction, Options, Dir, row),
    get_option(gap, Options, Gap, 0),
    get_option(justify, Options, Justify, start),
    get_option(align, Options, Align, stretch),
    get_option(wrap, Options, Wrap, false),
    get_option(padding, Options, Padding, 0),

    direction_style(Dir, DirStyle),
    gap_style(Gap, GapStyle),
    justify_style(Justify, JustifyStyle),
    align_style(Align, AlignStyle),
    wrap_style(Wrap, WrapStyle),
    padding_style(Padding, PaddingStyle),

    build_style_object([DirStyle, GapStyle, JustifyStyle, AlignStyle, WrapStyle, PaddingStyle], StyleObj),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ display: "flex", ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ChildrenCode, IndentStr]).

% Grid layout
generate_layout(grid, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(columns, Options, Cols, 1),
    get_option(gap, Options, Gap, 0),
    get_option(padding, Options, Padding, 0),

    format(atom(GridStyle), 'gridTemplateColumns: "repeat(~w, 1fr)"', [Cols]),
    gap_style(Gap, GapStyle),
    padding_style(Padding, PaddingStyle),

    build_style_object([GridStyle, GapStyle, PaddingStyle], StyleObj),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ display: "grid", ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ChildrenCode, IndentStr]).

% Scroll layout
generate_layout(scroll, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(direction, Options, Dir, vertical),
    get_option(padding, Options, Padding, 0),

    (Dir = vertical -> Overflow = 'overflowY: "auto"' ; Overflow = 'overflowX: "auto"'),
    padding_style(Padding, PaddingStyle),

    build_style_object([Overflow, PaddingStyle], StyleObj),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ChildrenCode, IndentStr]).

% Center layout
generate_layout(center, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(padding, Options, Padding, 0),
    padding_style(Padding, PaddingStyle),

    build_style_object([PaddingStyle], StyleObj),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ display: "flex", justifyContent: "center", alignItems: "center", ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ChildrenCode, IndentStr]).

% Wrap layout
generate_layout(wrap, Options, Children, Indent, GenOpts, Code) :- !,
    get_option(gap, Options, Gap, 0),
    gap_style(Gap, GapStyle),

    build_style_object([GapStyle], StyleObj),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ display: "flex", flexWrap: "wrap", ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ChildrenCode, IndentStr]).

% Default layout fallback
generate_layout(Type, _Options, Children, Indent, GenOpts, Code) :-
    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w{/* Layout: ~w */}~n~w<div>~n~w~w</div>~n',
           [IndentStr, Type, IndentStr, ChildrenCode, IndentStr]).

% ============================================================================
% CONTAINER GENERATION
% ============================================================================

%! generate_container(+Type, +Options, +Content, +Indent, +GenOpts, -Code) is det

% Panel container
generate_container(panel, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(background, Options, Bg, '#16213e'),
    get_option(padding, Options, Padding, 20),
    get_option(rounded, Options, Rounded, 5),

    format(atom(StyleObj), 'background: "~w", padding: ~w, borderRadius: ~w', [Bg, Padding, Rounded]),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ContentCode, IndentStr]).

% Card container
generate_container(card, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(padding, Options, Padding, 16),
    get_option(elevation, Options, Elevation, 2),

    shadow_style(Elevation, Shadow),
    format(atom(StyleObj), 'background: "#fff", padding: ~w, borderRadius: 8, ~w', [Padding, Shadow]),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ContentCode, IndentStr]).

% Scroll container
generate_container(scroll, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(max_height, Options, MaxHeight, 400),
    get_option(direction, Options, Dir, vertical),

    (Dir = vertical -> Overflow = 'overflowY: "auto"' ; Overflow = 'overflowX: "auto"'),
    format(atom(StyleObj), '~w, maxHeight: ~w', [Overflow, MaxHeight]),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ ~w }}>~n~w~w</div>~n',
           [IndentStr, StyleObj, ContentCode, IndentStr]).

% Section container
generate_container(section, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, ''),
    get_option(padding, Options, Padding, 0),

    padding_style(Padding, PaddingStyle),
    build_style_object([PaddingStyle], StyleObj),
    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    (Title \= '' ->
        format(atom(Code), '~w<section style={{ ~w }}>~n~w<h3>~w</h3>~n~w~w</section>~n',
               [IndentStr, StyleObj, NextIndentStr, Title, ContentCode, IndentStr])
    ;
        format(atom(Code), '~w<section style={{ ~w }}>~n~w~w</section>~n',
               [IndentStr, StyleObj, ContentCode, IndentStr])
    ).

% Modal container
generate_container(modal, Options, Content, Indent, GenOpts, Code) :- !,
    get_option(title, Options, Title, ''),
    get_option(on_close, Options, OnClose, ''),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code),
'~w<div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.5)", display: "flex", justifyContent: "center", alignItems: "center" }}>
~w<div style={{ background: "#fff", padding: 24, borderRadius: 8, minWidth: 300 }}>
~w<div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
~w<h3>~w</h3>
~w<button onClick={~w} style={{ background: "none", border: "none", cursor: "pointer" }}>&times;</button>
~w</div>
~w~w</div>
~w</div>~n',
           [IndentStr, NextIndentStr, NextIndentStr, NextIndentStr, Title,
            NextIndentStr, OnClose, NextIndentStr, ContentCode, NextIndentStr, IndentStr]).

% Default container fallback
generate_container(Type, _Options, Content, Indent, GenOpts, Code) :-
    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w{/* Container: ~w */}~n~w<div>~n~w~w</div>~n',
           [IndentStr, Type, IndentStr, ContentCode, IndentStr]).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%! generate_component(+Type, +Options, +Indent, +GenOpts, -Code) is det

% Text component
generate_component(text, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    get_option(style, Options, Style, ''),
    get_option(color, Options, Color, ''),

    build_style_props([color-Color], Style, StyleAttr),
    indent_string(Indent, IndentStr),

    (is_binding(Content) ->
        binding_to_jsx(Content, JsxBinding),
        format(atom(Code), '~w<span~w>{~w}</span>~n', [IndentStr, StyleAttr, JsxBinding])
    ;
        format(atom(Code), '~w<span~w>~w</span>~n', [IndentStr, StyleAttr, Content])
    ).

% Heading component
generate_component(heading, Options, Indent, _GenOpts, Code) :- !,
    get_option(level, Options, Level, 1),
    get_option(content, Options, Content, ''),

    indent_string(Indent, IndentStr),
    (is_binding(Content) ->
        binding_to_jsx(Content, JsxBinding),
        format(atom(Code), '~w<h~w>{~w}</h~w>~n', [IndentStr, Level, JsxBinding, Level])
    ;
        format(atom(Code), '~w<h~w>~w</h~w>~n', [IndentStr, Level, Content, Level])
    ).

% Label component
generate_component(label, Options, Indent, _GenOpts, Code) :- !,
    get_option(text, Options, Text, ''),
    get_option(for, Options, For, ''),

    indent_string(Indent, IndentStr),
    (For \= '' ->
        format(atom(Code), '~w<label htmlFor="~w">~w</label>~n', [IndentStr, For, Text])
    ;
        format(atom(Code), '~w<label>~w</label>~n', [IndentStr, Text])
    ).

% Link component
generate_component(link, Options, Indent, _GenOpts, Code) :- !,
    get_option(href, Options, Href, '#'),
    get_option(label, Options, Label, ''),
    get_option(on_click, Options, OnClick, ''),

    indent_string(Indent, IndentStr),
    (OnClick \= '' ->
        format(atom(Code), '~w<a href="~w" onClick={(e) => { e.preventDefault(); ~w(); }}>~w</a>~n',
               [IndentStr, Href, OnClick, Label])
    ;
        format(atom(Code), '~w<a href="~w">~w</a>~n', [IndentStr, Href, Label])
    ).

% Button component
generate_component(button, Options, Indent, _GenOpts, Code) :- !,
    get_option(label, Options, Label, 'Button'),
    get_option(on_click, Options, OnClick, ''),
    get_option(variant, Options, Variant, primary),
    get_option(disabled, Options, Disabled, false),
    get_option(loading, Options, Loading, false),

    variant_button_style(Variant, VariantStyle),
    indent_string(Indent, IndentStr),

    (OnClick \= '' -> format(atom(ClickAttr), ' onClick={~w}', [OnClick]) ; ClickAttr = ''),
    (Disabled \= false -> format(atom(DisabledAttr), ' disabled={~w}', [Disabled]) ; DisabledAttr = ''),

    (Loading \= false ->
        format(atom(Code), '~w<button~w~w style={{ ~w }}>{~w ? "Loading..." : "~w"}</button>~n',
               [IndentStr, ClickAttr, DisabledAttr, VariantStyle, Loading, Label])
    ;
        format(atom(Code), '~w<button~w~w style={{ ~w }}>~w</button>~n',
               [IndentStr, ClickAttr, DisabledAttr, VariantStyle, Label])
    ).

% Icon button component
generate_component(icon_button, Options, Indent, _GenOpts, Code) :- !,
    get_option(icon, Options, Icon, ''),
    get_option(on_click, Options, OnClick, ''),
    get_option(aria_label, Options, AriaLabel, ''),

    indent_string(Indent, IndentStr),
    (OnClick \= '' -> format(atom(ClickAttr), ' onClick={~w}', [OnClick]) ; ClickAttr = ''),
    format(atom(Code), '~w<button~w aria-label="~w" style={{ background: "none", border: "none", cursor: "pointer" }}>~w</button>~n',
           [IndentStr, ClickAttr, AriaLabel, Icon]).

% Text input component
generate_component(text_input, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(type, Options, Type, text),
    get_option(placeholder, Options, Placeholder, ''),
    get_option(label, Options, Label, ''),
    get_option(disabled, Options, Disabled, false),

    indent_string(Indent, IndentStr),
    (Disabled \= false -> format(atom(DisabledAttr), ' disabled={~w}', [Disabled]) ; DisabledAttr = ''),

    % React uses value + onChange instead of v-model
    capitalize_first(Bind, CapBind),
    format(atom(InputAttrs), 'type="~w" value={~w} onChange={(e) => set~w(e.target.value)} placeholder="~w"~w',
           [Type, Bind, CapBind, Placeholder, DisabledAttr]),

    (Label \= '' ->
        format(atom(Code),
'~w<div className="form-group">
~w  <label>~w</label>
~w  <input ~w style={{ width: "100%", padding: 10, border: "1px solid #ccc", borderRadius: 5 }} />
~w</div>~n',
               [IndentStr, IndentStr, Label, IndentStr, InputAttrs, IndentStr])
    ;
        format(atom(Code), '~w<input ~w style={{ width: "100%", padding: 10, border: "1px solid #ccc", borderRadius: 5 }} />~n',
               [IndentStr, InputAttrs])
    ).

% Textarea component
generate_component(textarea, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(placeholder, Options, Placeholder, ''),
    get_option(rows, Options, Rows, 4),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),

    capitalize_first(Bind, CapBind),
    format(atom(TextareaAttrs), 'value={~w} onChange={(e) => set~w(e.target.value)} placeholder="~w" rows={~w}',
           [Bind, CapBind, Placeholder, Rows]),

    (Label \= '' ->
        format(atom(Code),
'~w<div className="form-group">
~w  <label>~w</label>
~w  <textarea ~w style={{ width: "100%", padding: 10, border: "1px solid #ccc", borderRadius: 5 }} />
~w</div>~n',
               [IndentStr, IndentStr, Label, IndentStr, TextareaAttrs, IndentStr])
    ;
        format(atom(Code), '~w<textarea ~w style={{ width: "100%", padding: 10 }} />~n',
               [IndentStr, TextareaAttrs])
    ).

% Checkbox component
generate_component(checkbox, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),
    capitalize_first(Bind, CapBind),
    format(atom(Code), '~w<label style={{ display: "flex", alignItems: "center", gap: 8 }}><input type="checkbox" checked={~w} onChange={(e) => set~w(e.target.checked)} /> ~w</label>~n',
           [IndentStr, Bind, CapBind, Label]).

% Select component
generate_component(select, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(options, Options, Opts, []),
    get_option(placeholder, Options, Placeholder, 'Select...'),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),

    (is_binding(Opts) ->
        binding_to_jsx(Opts, JsxOpts),
        format(atom(OptionsCode), '{~w.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}', [JsxOpts])
    ;
        generate_static_options(Opts, OptionsCode)
    ),

    capitalize_first(Bind, CapBind),
    format(atom(SelectAttrs), 'value={~w} onChange={(e) => set~w(e.target.value)}', [Bind, CapBind]),

    (Label \= '' ->
        format(atom(Code),
'~w<div className="form-group">
~w  <label>~w</label>
~w  <select ~w style={{ width: "100%", padding: 10, border: "1px solid #ccc", borderRadius: 5 }}>
~w    <option value="" disabled>~w</option>
~w    ~w
~w  </select>
~w</div>~n',
               [IndentStr, IndentStr, Label, IndentStr, SelectAttrs, IndentStr, Placeholder, IndentStr, OptionsCode, IndentStr, IndentStr])
    ;
        format(atom(Code), '~w<select ~w style={{ width: "100%", padding: 10 }}><option value="" disabled>~w</option>~w</select>~n',
               [IndentStr, SelectAttrs, Placeholder, OptionsCode])
    ).

% Switch component
generate_component(switch, Options, Indent, _GenOpts, Code) :- !,
    get_option(bind, Options, Bind, ''),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),
    capitalize_first(Bind, CapBind),
    format(atom(Code),
'~w<label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
~w  <input type="checkbox" checked={~w} onChange={(e) => set~w(e.target.checked)} style={{ display: "none" }} />
~w  <span style={{ width: 40, height: 20, background: ~w ? "#4ade80" : "#ccc", borderRadius: 10, position: "relative", transition: "0.3s" }}>
~w    <span style={{ position: "absolute", top: 2, left: ~w ? 22 : 2, width: 16, height: 16, background: "#fff", borderRadius: "50%", transition: "0.3s" }} />
~w  </span>
~w  ~w
~w</label>~n',
           [IndentStr, IndentStr, Bind, CapBind, IndentStr, Bind, IndentStr, Bind, IndentStr, IndentStr, Label, IndentStr]).

% Tabs component
generate_component(tabs, Options, Indent, _GenOpts, Code) :- !,
    get_option(items, Options, Items, []),
    get_option(active, Options, Active, ''),
    get_option(on_change, Options, OnChange, ''),

    indent_string(Indent, IndentStr),

    capitalize_first(Active, CapActive),
    (is_binding(Items) ->
        binding_to_jsx(Items, JsxItems),
        (OnChange \= '' ->
            format(atom(TabCode), '{~w.map(item => <button key={item} onClick={() => { set~w(item); ~w(item); }} className={~w === item ? "tab active" : "tab"}>{item}</button>)}',
                   [JsxItems, CapActive, OnChange, Active])
        ;
            format(atom(TabCode), '{~w.map(item => <button key={item} onClick={() => set~w(item)} className={~w === item ? "tab active" : "tab"}>{item}</button>)}',
                   [JsxItems, CapActive, Active])
        )
    ;
        generate_static_tabs(Items, Active, OnChange, TabCode)
    ),

    format(atom(Code), '~w<div className="tabs" style={{ display: "flex", gap: 5 }}>~w</div>~n', [IndentStr, TabCode]).

% Spinner component
generate_component(spinner, Options, Indent, _GenOpts, Code) :- !,
    get_option(size, Options, Size, 24),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ width: ~w, height: ~w, border: "3px solid #f3f3f3", borderTop: "3px solid #3498db", borderRadius: "50%", animation: "spin 1s linear infinite" }} />~n',
           [IndentStr, Size, Size]).

% Progress component
generate_component(progress, Options, Indent, _GenOpts, Code) :- !,
    get_option(value, Options, Value, 0),
    get_option(max, Options, Max, 100),

    indent_string(Indent, IndentStr),
    (is_binding(Value) ->
        binding_to_jsx(Value, JsxValue),
        format(atom(Code), '~w<div style={{ width: "100%", height: 8, background: "#eee", borderRadius: 4, overflow: "hidden" }}><div style={{ width: `${(~w / ~w * 100)}%`, height: "100%", background: "#3498db" }} /></div>~n',
               [IndentStr, JsxValue, Max])
    ;
        Percent is Value / Max * 100,
        format(atom(Code), '~w<div style={{ width: "100%", height: 8, background: "#eee", borderRadius: 4, overflow: "hidden" }}><div style={{ width: "~w%", height: "100%", background: "#3498db" }} /></div>~n',
               [IndentStr, Percent])
    ).

% Badge component
generate_component(badge, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),
    get_option(variant, Options, Variant, info),

    variant_badge_style(Variant, BadgeStyle),
    indent_string(Indent, IndentStr),

    (is_binding(Content) ->
        binding_to_jsx(Content, JsxContent),
        format(atom(Code), '~w<span style={{ ~w }}>{~w}</span>~n', [IndentStr, BadgeStyle, JsxContent])
    ;
        format(atom(Code), '~w<span style={{ ~w }}>~w</span>~n', [IndentStr, BadgeStyle, Content])
    ).

% Divider component
generate_component(divider, Options, Indent, _GenOpts, Code) :- !,
    get_option(margin, Options, Margin, 16),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<hr style={{ border: "none", borderTop: "1px solid #eee", margin: "~wpx 0" }} />~n', [IndentStr, Margin]).

% Spacer component
generate_component(spacer, Options, Indent, _GenOpts, Code) :- !,
    get_option(size, Options, Size, 16),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ height: ~w }} />~n', [IndentStr, Size]).

% Image component
generate_component(image, Options, Indent, _GenOpts, Code) :- !,
    get_option(src, Options, Src, ''),
    get_option(alt, Options, Alt, ''),
    get_option(width, Options, Width, ''),
    get_option(height, Options, Height, ''),

    indent_string(Indent, IndentStr),
    build_dimension_style_jsx(Width, Height, DimStyle),

    (is_binding(Src) ->
        binding_to_jsx(Src, JsxSrc),
        format(atom(Code), '~w<img src={~w} alt="~w"~w />~n', [IndentStr, JsxSrc, Alt, DimStyle])
    ;
        format(atom(Code), '~w<img src="~w" alt="~w"~w />~n', [IndentStr, Src, Alt, DimStyle])
    ).

% Avatar component
generate_component(avatar, Options, Indent, _GenOpts, Code) :- !,
    get_option(src, Options, Src, ''),
    get_option(name, Options, Name, ''),
    get_option(size, Options, Size, 40),

    indent_string(Indent, IndentStr),
    (   Src \= ''
    ->  format(atom(Code), '~w<img src="~w" alt="~w" style={{ width: ~w, height: ~w, borderRadius: "50%", objectFit: "cover" }} />~n',
               [IndentStr, Src, Name, Size, Size])
    ;   format(atom(Code), '~w<div style={{ width: ~w, height: ~w, borderRadius: "50%", background: "#3498db", display: "flex", justifyContent: "center", alignItems: "center", color: "#fff", fontWeight: "bold", overflow: "hidden" }}>~w</div>~n',
               [IndentStr, Size, Size, Name])
    ).

% Icon component
generate_component(icon, Options, Indent, _GenOpts, Code) :- !,
    get_option(name, Options, Name, ''),
    get_option(size, Options, Size, 24),

    indent_string(Indent, IndentStr),
    (is_binding(Name) ->
        binding_to_jsx(Name, JsxName),
        format(atom(Code), '~w<span style={{ fontSize: ~w }}>{~w}</span>~n', [IndentStr, Size, JsxName])
    ;
        format(atom(Code), '~w<span style={{ fontSize: ~w }}>~w</span>~n', [IndentStr, Size, Name])
    ).

% Code component (inline code display)
generate_component(code, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),

    indent_string(Indent, IndentStr),
    (is_binding(Content) ->
        binding_to_jsx(Content, JsxContent),
        format(atom(Code), '~w<code style={{ background: "#1a1a2e", padding: "4px 8px", borderRadius: 3, fontFamily: "monospace" }}>{~w}</code>~n', [IndentStr, JsxContent])
    ;
        format(atom(Code), '~w<code style={{ background: "#1a1a2e", padding: "4px 8px", borderRadius: 3, fontFamily: "monospace" }}>~w</code>~n', [IndentStr, Content])
    ).

% Pre component (preformatted text)
generate_component(pre, Options, Indent, _GenOpts, Code) :- !,
    get_option(content, Options, Content, ''),

    indent_string(Indent, IndentStr),
    (is_binding(Content) ->
        binding_to_jsx(Content, JsxContent),
        format(atom(Code), '~w<pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{~w}</pre>~n', [IndentStr, JsxContent])
    ;
        format(atom(Code), '~w<pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>~w</pre>~n', [IndentStr, Content])
    ).

% Alert component
generate_component(alert, Options, Indent, _GenOpts, Code) :- !,
    get_option(message, Options, Message, ''),
    get_option(variant, Options, Variant, info),

    variant_alert_style(Variant, AlertStyle),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style={{ ~w }}>~w</div>~n', [IndentStr, AlertStyle, Message]).

% Default component fallback
generate_component(Type, Options, Indent, _GenOpts, Code) :-
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w{/* Component: ~w ~w */}~n', [IndentStr, Type, Options]).

% ============================================================================
% CONDITIONAL GENERATION
% ============================================================================

generate_conditional(Condition, Content, Indent, GenOpts, Code) :- !,
    condition_to_jsx(Condition, JsxCond),
    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w{(~w) && (~n~w~w)}~n',
           [IndentStr, JsxCond, ContentCode, IndentStr]).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

%! generate_children(+Children, +Indent, +Options, -Code) is det
generate_children([], _Indent, _Options, '') :- !.
generate_children([Child|Rest], Indent, Options, Code) :-
    NextIndent is Indent + 1,
    generate_node(Child, NextIndent, Options, ChildCode),
    generate_children(Rest, Indent, Options, RestCode),
    atom_concat(ChildCode, RestCode, Code).

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

%! capitalize_first(+Atom, -Capitalized) is det
%  Capitalize the first letter of an atom.
capitalize_first(Atom, Capitalized) :-
    atom_codes(Atom, [First|Rest]),
    (   First >= 0'a, First =< 0'z
    ->  Upper is First - 32,
        atom_codes(Capitalized, [Upper|Rest])
    ;   Capitalized = Atom
    ).

% Style helpers (React uses camelCase)
direction_style(row, 'flexDirection: "row"').
direction_style(column, 'flexDirection: "column"').

spacing_style(0, '') :- !.
spacing_style(N, Style) :- format(atom(Style), 'gap: ~w', [N]).

gap_style(0, '') :- !.
gap_style(N, Style) :- format(atom(Style), 'gap: ~w', [N]).

padding_style(0, '') :- !.
padding_style(N, Style) :- format(atom(Style), 'padding: ~w', [N]).

align_style(start, 'alignItems: "flex-start"').
align_style(center, 'alignItems: "center"').
align_style(end, 'alignItems: "flex-end"').
align_style(stretch, 'alignItems: "stretch"').
align_style(_, '').

justify_style(start, 'justifyContent: "flex-start"').
justify_style(center, 'justifyContent: "center"').
justify_style(end, 'justifyContent: "flex-end"').
justify_style(between, 'justifyContent: "space-between"').
justify_style(around, 'justifyContent: "space-around"').
justify_style(evenly, 'justifyContent: "space-evenly"').
justify_style(_, '').

wrap_style(true, 'flexWrap: "wrap"').
wrap_style(false, '').
wrap_style(_, '').

shadow_style(0, '') :- !.
shadow_style(1, 'boxShadow: "0 1px 3px rgba(0,0,0,0.12)"').
shadow_style(2, 'boxShadow: "0 3px 6px rgba(0,0,0,0.15)"').
shadow_style(3, 'boxShadow: "0 10px 20px rgba(0,0,0,0.19)"').
shadow_style(_, 'boxShadow: "0 3px 6px rgba(0,0,0,0.15)"').

% Build style object from list of styles
build_style_object(Styles, Result) :-
    exclude(==(''), Styles, NonEmpty),
    atomic_list_concat(NonEmpty, ', ', Result).

% Variant styles (React format)
variant_button_style(primary, 'padding: "10px 20px", background: "#e94560", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5, fontWeight: "bold"').
variant_button_style(secondary, 'padding: "10px 20px", background: "#16213e", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5').
variant_button_style(danger, 'padding: "10px 20px", background: "#dc3545", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5').
variant_button_style(_, 'padding: "10px 20px", background: "#e94560", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5').

variant_badge_style(info, 'padding: "2px 8px", background: "#3498db", color: "#fff", borderRadius: 3, fontSize: 12').
variant_badge_style(success, 'padding: "2px 8px", background: "#4ade80", color: "#000", borderRadius: 3, fontSize: 12').
variant_badge_style(warning, 'padding: "2px 8px", background: "#fbbf24", color: "#000", borderRadius: 3, fontSize: 12').
variant_badge_style(error, 'padding: "2px 8px", background: "#ff6b6b", color: "#fff", borderRadius: 3, fontSize: 12').
variant_badge_style(_, 'padding: "2px 8px", background: "#ccc", borderRadius: 3, fontSize: 12').

variant_alert_style(info, 'padding: "12px 16px", background: "#d1ecf1", color: "#0c5460", borderRadius: 5').
variant_alert_style(success, 'padding: "12px 16px", background: "#d4edda", color: "#155724", borderRadius: 5').
variant_alert_style(warning, 'padding: "12px 16px", background: "#fff3cd", color: "#856404", borderRadius: 5').
variant_alert_style(error, 'padding: "12px 16px", background: "#f8d7da", color: "#721c24", borderRadius: 5').
variant_alert_style(_, 'padding: "12px 16px", background: "#eee", borderRadius: 5').

% Binding detection
is_binding(var(_)) :- !.
is_binding(bind(_)) :- !.
is_binding(Term) :- atom(Term), atom_codes(Term, [C|_]), C \= 34, C \= 39.  % Not starting with quote

binding_to_jsx(var(X), X) :- !.
binding_to_jsx(bind(X), X) :- !.
binding_to_jsx(X, X) :- atom(X).

% Condition conversion for JSX
condition_to_jsx(var(X), X) :- !.
condition_to_jsx(Cond, Cond) :- atom(Cond), !.
condition_to_jsx(not(C), JsxCond) :- !,
    condition_to_jsx(C, VC),
    format(atom(JsxCond), '!~w', [VC]).
condition_to_jsx(and(A, B), JsxCond) :- !,
    condition_to_jsx(A, VA),
    condition_to_jsx(B, VB),
    format(atom(JsxCond), '(~w && ~w)', [VA, VB]).
condition_to_jsx(or(A, B), JsxCond) :- !,
    condition_to_jsx(A, VA),
    condition_to_jsx(B, VB),
    format(atom(JsxCond), '(~w || ~w)', [VA, VB]).
condition_to_jsx(eq(A, B), JsxCond) :- !,
    condition_to_jsx(A, VA),
    condition_to_jsx(B, VB),
    format(atom(JsxCond), '(~w === ~w)', [VA, VB]).
condition_to_jsx(not_eq(A, B), JsxCond) :- !,
    condition_to_jsx(A, VA),
    condition_to_jsx(B, VB),
    format(atom(JsxCond), '(~w !== ~w)', [VA, VB]).
condition_to_jsx(empty(C), JsxCond) :- !,
    condition_to_jsx(C, VC),
    format(atom(JsxCond), '(!~w || ~w.length === 0)', [VC, VC]).
condition_to_jsx([H|T], JsxCond) :- !,
    condition_to_jsx(H, VH),
    condition_to_jsx(T, VT),
    format(atom(JsxCond), '(~w && ~w)', [VH, VT]).
condition_to_jsx([], 'true') :- !.
% Fallback for unknown conditions - convert to string
condition_to_jsx(Term, Jsx) :-
    term_to_atom(Term, Jsx).

items_to_jsx(var(X), X) :- !.
items_to_jsx(Items, Items) :- atom(Items), !.
items_to_jsx(Items, JsxItems) :- is_list(Items), !, term_to_atom(Items, JsxItems).
items_to_jsx(Term, Jsx) :- term_to_atom(Term, Jsx).

% Build style props
build_style_props([], '', '') :- !.
build_style_props(Pairs, BaseStyle, StyleAttr) :-
    findall(S, (member(K-V, Pairs), V \= '', format(atom(S), '~w: "~w"', [K, V])), Styles),
    (BaseStyle \= '' -> append(Styles, [BaseStyle], AllStyles) ; AllStyles = Styles),
    (AllStyles \= [] ->
        atomic_list_concat(AllStyles, ', ', StyleStr),
        format(atom(StyleAttr), ' style={{ ~w }}', [StyleStr])
    ;
        StyleAttr = ''
    ).

build_dimension_style_jsx('', '', '') :- !.
build_dimension_style_jsx(W, '', Style) :- W \= '', !, format(atom(Style), ' style={{ width: "~w" }}', [W]).
build_dimension_style_jsx('', H, Style) :- H \= '', !, format(atom(Style), ' style={{ height: "~w" }}', [H]).
build_dimension_style_jsx(W, H, Style) :- W \= '', H \= '', format(atom(Style), ' style={{ width: "~w", height: "~w" }}', [W, H]).

generate_static_options([], '') :- !.
generate_static_options([Opt|Rest], Code) :-
    (Opt = opt(Value, Label) ->
        format(atom(OptCode), '<option value="~w">~w</option>', [Value, Label])
    ;
        format(atom(OptCode), '<option value="~w">~w</option>', [Opt, Opt])
    ),
    generate_static_options(Rest, RestCode),
    atom_concat(OptCode, RestCode, Code).

generate_static_tabs([], _Active, _OnChange, '') :- !.
generate_static_tabs([Tab|Rest], Active, OnChange, Code) :-
    capitalize_first(Active, CapActive),
    (OnChange \= '' ->
        format(atom(TabCode), '<button onClick={() => { set~w("~w"); ~w("~w"); }} className={~w === "~w" ? "tab active" : "tab"}>~w</button>',
               [CapActive, Tab, OnChange, Tab, Active, Tab, Tab])
    ;
        format(atom(TabCode), '<button onClick={() => set~w("~w")} className={~w === "~w" ? "tab active" : "tab"}>~w</button>',
               [CapActive, Tab, Active, Tab, Tab])
    ),
    generate_static_tabs(Rest, Active, OnChange, RestCode),
    atom_concat(TabCode, RestCode, Code).

% ============================================================================
% FULL COMPONENT GENERATION
% ============================================================================

%! generate_react_component(+UISpec, +Options, -Component) is det
%  Generate a complete React functional component.
generate_react_component(UISpec, Options, Component) :-
    get_option(name, Options, Name, 'GeneratedComponent'),
    generate_react_template(UISpec, Options, Template),
    format(atom(Component),
'import React, { useState } from "react";

function ~w() {
  // TODO: Extract state from UI spec

  return (
~w  );
}

export default ~w;
', [Name, Template, Name]).

% ============================================================================
% TESTING
% ============================================================================

test_react_generator :-
    format('~n=== React Generator Tests ===~n~n'),

    % Test 1: Simple text
    format('Test 1: Simple text component...~n'),
    generate_react_template(component(text, [content("Hello World")]), [], T1),
    format('  Output: ~w~n', [T1]),

    % Test 2: Button
    format('~nTest 2: Button component...~n'),
    generate_react_template(component(button, [label("Click Me"), on_click(handleClick)]), [], T2),
    format('  Output: ~w~n', [T2]),

    % Test 3: Stack layout
    format('~nTest 3: Stack layout...~n'),
    UI3 = layout(stack, [spacing(16)], [
        component(heading, [level(1), content("Title")]),
        component(text, [content("Description")])
    ]),
    generate_react_template(UI3, [], T3),
    format('  Output:~n~w~n', [T3]),

    % Test 4: Form with inputs
    format('~nTest 4: Form with inputs...~n'),
    UI4 = layout(stack, [spacing(12)], [
        component(text_input, [label("Email"), type(email), bind(email), placeholder("Enter email")]),
        component(text_input, [label("Password"), type(password), bind(password)]),
        component(button, [label("Submit"), on_click(submit), variant(primary)])
    ]),
    generate_react_template(UI4, [], T4),
    format('  Output:~n~w~n', [T4]),

    % Test 5: Conditional
    format('~nTest 5: Conditional rendering...~n'),
    UI5 = when(isLoggedIn, component(text, [content("Welcome!")])),
    generate_react_template(UI5, [], T5),
    format('  Output: ~w~n', [T5]),

    % Test 6: Foreach
    format('~nTest 6: Foreach iteration...~n'),
    UI6 = foreach(items, item, component(text, [content(item)])),
    generate_react_template(UI6, [], T6),
    format('  Output: ~w~n', [T6]),

    % Test 7: Nested containers
    format('~nTest 7: Nested containers...~n'),
    UI7 = container(panel, [padding(20), background('#1a1a2e')], [
        layout(stack, [spacing(8)], [
            component(heading, [level(2), content("Panel Title")]),
            component(text, [content("Panel content goes here")])
        ])
    ]),
    generate_react_template(UI7, [], T7),
    format('  Output:~n~w~n', [T7]),

    % Test 8: Full component generation
    format('~nTest 8: Full component generation...~n'),
    UI8 = layout(stack, [spacing(16)], [
        component(heading, [level(1), content("My App")]),
        component(button, [label("Click"), on_click(handleClick)])
    ]),
    generate_react_component(UI8, [name('MyComponent')], C8),
    format('  Output:~n~w~n', [C8]),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('React Generator module loaded~n')
), now).
