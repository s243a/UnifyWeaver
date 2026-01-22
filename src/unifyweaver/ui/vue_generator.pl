% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% vue_generator.pl - Generate Vue.js code from UI primitives
%
% Compiles declarative UI specifications to Vue 3 Single File Components
% or inline template code.
%
% Usage:
%   use_module('src/unifyweaver/ui/vue_generator').
%   UI = layout(stack, [spacing(16)], [
%       component(text, [content("Hello")]),
%       component(button, [label("Click"), on_click(submit)])
%   ]),
%   generate_vue_template(UI, Template).

:- module(vue_generator, [
    % Template generation
    generate_vue_template/2,      % generate_vue_template(+UISpec, -Template)
    generate_vue_template/3,      % generate_vue_template(+UISpec, +Options, -Template)

    % Full SFC generation
    generate_vue_sfc/3,           % generate_vue_sfc(+UISpec, +Options, -SFC)

    % Script generation
    generate_vue_script/3,        % generate_vue_script(+UISpec, +Options, -Script)

    % Style generation
    generate_vue_styles/3,        % generate_vue_styles(+UISpec, +Options, -Styles)

    % Testing
    test_vue_generator/0
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

% ============================================================================
% TEMPLATE GENERATION
% ============================================================================

%! generate_vue_template(+UISpec, -Template) is det
%  Generate Vue template from UI specification.
generate_vue_template(UISpec, Template) :-
    generate_vue_template(UISpec, [], Template).

%! generate_vue_template(+UISpec, +Options, -Template) is det
%  Generate Vue template with options.
%  Options:
%    indent(N) - Starting indentation level
%    style(tailwind|css|inline) - Styling approach
generate_vue_template(UISpec, Options, Template) :-
    get_option(indent, Options, Indent, 0),
    generate_node(UISpec, Indent, Options, Template).

% ============================================================================
% NODE GENERATION
% ============================================================================

%! generate_node(+Node, +Indent, +Options, -Code) is det
%  Generate Vue code for any UI node.

% Layout nodes
generate_node(layout(Type, LayoutOpts, Children), Indent, Options, Code) :-
    generate_layout(Type, LayoutOpts, Children, Indent, Options, Code), !.

% Container nodes
generate_node(container(Type, ContainerOpts, Content), Indent, Options, Code) :-
    generate_container(Type, ContainerOpts, Content, Indent, Options, Code), !.

% Component nodes
generate_node(component(Type, CompOpts), Indent, Options, Code) :-
    generate_component(Type, CompOpts, Indent, Options, Code), !.

% Conditional: when
generate_node(when(Condition, Content), Indent, Options, Code) :-
    generate_conditional(Condition, Content, Indent, Options, Code), !.

% Conditional: unless
generate_node(unless(Condition, Content), Indent, Options, Code) :-
    condition_to_vue(Condition, VueCond),
    format(atom(NegCond), '!(~w)', [VueCond]),
    generate_node(Content, Indent, Options, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<template v-if="~w">~n~w~w</template>',
           [IndentStr, NegCond, ContentCode, IndentStr]), !.

% Foreach iteration
generate_node(foreach(Items, Var, Template), Indent, Options, Code) :-
    items_to_vue(Items, VueItems),
    generate_node(Template, Indent, Options, TemplateCode),
    indent_string(Indent, IndentStr),
    % Wrap template content in a div with v-for
    format(atom(Code), '~w<div v-for="~w in ~w" :key="~w">~n~w~w</div>~n',
           [IndentStr, Var, VueItems, Var, TemplateCode, IndentStr]), !.

% Pattern usage (placeholder - patterns should be expanded before generation)
generate_node(use_pattern(Name, Args), Indent, _Options, Code) :-
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<!-- Pattern: ~w(~w) -->~n', [IndentStr, Name, Args]), !.

% List of nodes
generate_node([], _Indent, _Options, '') :- !.
generate_node([H|T], Indent, Options, Code) :-
    generate_node(H, Indent, Options, HCode),
    generate_node(T, Indent, Options, TCode),
    atom_concat(HCode, TCode, Code), !.

% Fallback
generate_node(Node, Indent, _Options, Code) :-
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<!-- Unknown node: ~w -->~n', [IndentStr, Node]).

% ============================================================================
% LAYOUT GENERATION
% ============================================================================

%! generate_layout(+Type, +Options, +Children, +Indent, +GenOpts, -Code) is det

% Stack layout -> div with flex-direction
generate_layout(stack, Options, Children, Indent, GenOpts, Code) :-
    get_option(direction, Options, Dir, column),
    get_option(spacing, Options, Spacing, 0),
    get_option(align, Options, Align, stretch),
    get_option(padding, Options, Padding, 0),

    direction_style(Dir, DirStyle),
    spacing_style(Spacing, SpacingStyle),
    align_style(Align, AlignStyle),
    padding_style(Padding, PaddingStyle),

    atomic_list_concat([DirStyle, SpacingStyle, AlignStyle, PaddingStyle], '; ', StyleStr),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="display: flex; ~w">~n~w~w</div>~n',
           [IndentStr, StyleStr, ChildrenCode, IndentStr]).

% Flex layout
generate_layout(flex, Options, Children, Indent, GenOpts, Code) :-
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

    atomic_list_concat([DirStyle, GapStyle, JustifyStyle, AlignStyle, WrapStyle, PaddingStyle], '; ', StyleStr),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="display: flex; ~w">~n~w~w</div>~n',
           [IndentStr, StyleStr, ChildrenCode, IndentStr]).

% Grid layout
generate_layout(grid, Options, Children, Indent, GenOpts, Code) :-
    get_option(columns, Options, Cols, 1),
    get_option(gap, Options, Gap, 0),
    get_option(padding, Options, Padding, 0),

    format(atom(GridStyle), 'grid-template-columns: repeat(~w, 1fr)', [Cols]),
    gap_style(Gap, GapStyle),
    padding_style(Padding, PaddingStyle),

    atomic_list_concat([GridStyle, GapStyle, PaddingStyle], '; ', StyleStr),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="display: grid; ~w">~n~w~w</div>~n',
           [IndentStr, StyleStr, ChildrenCode, IndentStr]).

% Scroll layout
generate_layout(scroll, Options, Children, Indent, GenOpts, Code) :-
    get_option(direction, Options, Dir, vertical),
    get_option(padding, Options, Padding, 0),

    (Dir = vertical -> Overflow = 'overflow-y: auto' ; Overflow = 'overflow-x: auto'),
    padding_style(Padding, PaddingStyle),

    atomic_list_concat([Overflow, PaddingStyle], '; ', StyleStr),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="~w">~n~w~w</div>~n',
           [IndentStr, StyleStr, ChildrenCode, IndentStr]).

% Center layout
generate_layout(center, Options, Children, Indent, GenOpts, Code) :-
    get_option(padding, Options, Padding, 0),
    padding_style(Padding, PaddingStyle),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="display: flex; justify-content: center; align-items: center; ~w">~n~w~w</div>~n',
           [IndentStr, PaddingStyle, ChildrenCode, IndentStr]).

% Positioned layout
generate_layout(positioned, Options, Children, Indent, GenOpts, Code) :-
    get_option(position, Options, Pos, relative),
    position_styles(Options, PosStyles),

    format(atom(StyleStr), 'position: ~w; ~w', [Pos, PosStyles]),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="~w">~n~w~w</div>~n',
           [IndentStr, StyleStr, ChildrenCode, IndentStr]).

% Wrap layout
generate_layout(wrap, Options, Children, Indent, GenOpts, Code) :-
    get_option(gap, Options, Gap, 0),
    gap_style(Gap, GapStyle),

    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="display: flex; flex-wrap: wrap; ~w">~n~w~w</div>~n',
           [IndentStr, GapStyle, ChildrenCode, IndentStr]).

% Default layout fallback
generate_layout(Type, _Options, Children, Indent, GenOpts, Code) :-
    generate_children(Children, Indent, GenOpts, ChildrenCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<!-- Layout: ~w -->~n~w<div>~n~w~w</div>~n',
           [IndentStr, Type, IndentStr, ChildrenCode, IndentStr]).

% ============================================================================
% CONTAINER GENERATION
% ============================================================================

%! generate_container(+Type, +Options, +Content, +Indent, +GenOpts, -Code) is det

% Panel container
generate_container(panel, Options, Content, Indent, GenOpts, Code) :-
    get_option(background, Options, Bg, '#16213e'),
    get_option(padding, Options, Padding, 20),
    get_option(rounded, Options, Rounded, 5),

    format(atom(StyleStr), 'background: ~w; padding: ~wpx; border-radius: ~wpx', [Bg, Padding, Rounded]),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="~w">~n~w~w</div>~n',
           [IndentStr, StyleStr, ContentCode, IndentStr]).

% Card container
generate_container(card, Options, Content, Indent, GenOpts, Code) :-
    get_option(padding, Options, Padding, 16),
    get_option(elevation, Options, Elevation, 2),

    shadow_style(Elevation, Shadow),
    format(atom(StyleStr), 'background: #fff; padding: ~wpx; border-radius: 8px; ~w', [Padding, Shadow]),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="~w">~n~w~w</div>~n',
           [IndentStr, StyleStr, ContentCode, IndentStr]).

% Section container
generate_container(section, Options, Content, Indent, GenOpts, Code) :-
    get_option(title, Options, Title, ''),
    get_option(padding, Options, Padding, 0),

    padding_style(Padding, PaddingStyle),
    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    (Title \= '' ->
        format(atom(Code), '~w<section style="~w">~n~w<h3>~w</h3>~n~w~w</section>~n',
               [IndentStr, PaddingStyle, NextIndentStr, Title, ContentCode, IndentStr])
    ;
        format(atom(Code), '~w<section style="~w">~n~w~w</section>~n',
               [IndentStr, PaddingStyle, ContentCode, IndentStr])
    ).

% Modal container
generate_container(modal, Options, Content, Indent, GenOpts, Code) :-
    get_option(title, Options, Title, ''),
    get_option(on_close, Options, OnClose, ''),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    format(atom(CloseHandler), '@click="~w"', [OnClose]),

    format(atom(Code),
'~w<div class="modal-backdrop" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center;">
~w<div class="modal" style="background: #fff; padding: 24px; border-radius: 8px; min-width: 300px;">
~w<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
~w<h3>~w</h3>
~w<button ~w style="background: none; border: none; cursor: pointer;">&times;</button>
~w</div>
~w~w</div>
~w</div>~n',
           [IndentStr, NextIndentStr, NextIndentStr, NextIndentStr, Title,
            NextIndentStr, CloseHandler, NextIndentStr, ContentCode, NextIndentStr, IndentStr]).

% Collapse container
generate_container(collapse, Options, Content, Indent, GenOpts, Code) :-
    get_option(header, Options, Header, 'Toggle'),
    get_option(expanded, Options, Expanded, false),

    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    indent_string(NextIndent, NextIndentStr),

    format(atom(Code),
'~w<div class="collapse">
~w<button @click="~w = !~w" style="width: 100%; text-align: left; padding: 10px; cursor: pointer;">~w</button>
~w<div v-show="~w">
~w~w</div>
~w</div>~n',
           [IndentStr, NextIndentStr, Expanded, Expanded, Header,
            NextIndentStr, Expanded, ContentCode, NextIndentStr, IndentStr]).

% Default container fallback
generate_container(Type, _Options, Content, Indent, GenOpts, Code) :-
    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<!-- Container: ~w -->~n~w<div>~n~w~w</div>~n',
           [IndentStr, Type, IndentStr, ContentCode, IndentStr]).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%! generate_component(+Type, +Options, +Indent, +GenOpts, -Code) is det

% Text component
generate_component(text, Options, Indent, _GenOpts, Code) :-
    get_option(content, Options, Content, ''),
    get_option(style, Options, Style, ''),
    get_option(color, Options, Color, ''),

    build_style_attr([color-Color], Style, StyleAttr),
    indent_string(Indent, IndentStr),

    (is_binding(Content) ->
        binding_to_vue(Content, VueBinding),
        format(atom(Code), '~w<span~w>{{ ~w }}</span>~n', [IndentStr, StyleAttr, VueBinding])
    ;
        format(atom(Code), '~w<span~w>~w</span>~n', [IndentStr, StyleAttr, Content])
    ).

% Heading component
generate_component(heading, Options, Indent, _GenOpts, Code) :-
    get_option(level, Options, Level, 1),
    get_option(content, Options, Content, ''),

    indent_string(Indent, IndentStr),
    (is_binding(Content) ->
        binding_to_vue(Content, VueBinding),
        format(atom(Code), '~w<h~w>{{ ~w }}</h~w>~n', [IndentStr, Level, VueBinding, Level])
    ;
        format(atom(Code), '~w<h~w>~w</h~w>~n', [IndentStr, Level, Content, Level])
    ).

% Label component
generate_component(label, Options, Indent, _GenOpts, Code) :-
    get_option(text, Options, Text, ''),
    get_option(for, Options, For, ''),

    indent_string(Indent, IndentStr),
    (For \= '' ->
        format(atom(Code), '~w<label for="~w">~w</label>~n', [IndentStr, For, Text])
    ;
        format(atom(Code), '~w<label>~w</label>~n', [IndentStr, Text])
    ).

% Link component
generate_component(link, Options, Indent, _GenOpts, Code) :-
    get_option(href, Options, Href, '#'),
    get_option(label, Options, Label, ''),
    get_option(on_click, Options, OnClick, ''),

    indent_string(Indent, IndentStr),
    (OnClick \= '' ->
        format(atom(Code), '~w<a href="~w" @click.prevent="~w">~w</a>~n', [IndentStr, Href, OnClick, Label])
    ;
        format(atom(Code), '~w<a href="~w">~w</a>~n', [IndentStr, Href, Label])
    ).

% Button component
generate_component(button, Options, Indent, _GenOpts, Code) :-
    get_option(label, Options, Label, 'Button'),
    get_option(on_click, Options, OnClick, ''),
    get_option(variant, Options, Variant, primary),
    get_option(disabled, Options, Disabled, false),
    get_option(loading, Options, Loading, false),

    variant_button_style(Variant, VariantStyle),
    indent_string(Indent, IndentStr),

    (OnClick \= '' -> format(atom(ClickAttr), ' @click="~w"', [OnClick]) ; ClickAttr = ''),
    (Disabled \= false -> format(atom(DisabledAttr), ' :disabled="~w"', [Disabled]) ; DisabledAttr = ''),

    (Loading \= false ->
        format(atom(Code), '~w<button~w~w style="~w">{{ ~w ? \'Loading...\' : \'~w\' }}</button>~n',
               [IndentStr, ClickAttr, DisabledAttr, VariantStyle, Loading, Label])
    ;
        format(atom(Code), '~w<button~w~w style="~w">~w</button>~n',
               [IndentStr, ClickAttr, DisabledAttr, VariantStyle, Label])
    ).

% Icon button component
generate_component(icon_button, Options, Indent, _GenOpts, Code) :-
    get_option(icon, Options, Icon, ''),
    get_option(on_click, Options, OnClick, ''),
    get_option(aria_label, Options, AriaLabel, ''),

    indent_string(Indent, IndentStr),
    (OnClick \= '' -> format(atom(ClickAttr), ' @click="~w"', [OnClick]) ; ClickAttr = ''),
    format(atom(Code), '~w<button~w aria-label="~w" style="background: none; border: none; cursor: pointer;">~w</button>~n',
           [IndentStr, ClickAttr, AriaLabel, Icon]).

% Text input component
generate_component(text_input, Options, Indent, _GenOpts, Code) :-
    get_option(bind, Options, Bind, ''),
    get_option(type, Options, Type, text),
    get_option(placeholder, Options, Placeholder, ''),
    get_option(label, Options, Label, ''),
    get_option(disabled, Options, Disabled, false),

    indent_string(Indent, IndentStr),
    (Disabled \= false -> format(atom(DisabledAttr), ' :disabled="~w"', [Disabled]) ; DisabledAttr = ''),

    (Label \= '' ->
        format(atom(Code),
'~w<div class="form-group">
~w  <label>~w</label>
~w  <input type="~w" v-model="~w" placeholder="~w"~w style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
~w</div>~n',
               [IndentStr, IndentStr, Label, IndentStr, Type, Bind, Placeholder, DisabledAttr, IndentStr])
    ;
        format(atom(Code), '~w<input type="~w" v-model="~w" placeholder="~w"~w style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">~n',
               [IndentStr, Type, Bind, Placeholder, DisabledAttr])
    ).

% Textarea component
generate_component(textarea, Options, Indent, _GenOpts, Code) :-
    get_option(bind, Options, Bind, ''),
    get_option(placeholder, Options, Placeholder, ''),
    get_option(rows, Options, Rows, 4),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),

    (Label \= '' ->
        format(atom(Code),
'~w<div class="form-group">
~w  <label>~w</label>
~w  <textarea v-model="~w" placeholder="~w" rows="~w" style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"></textarea>
~w</div>~n',
               [IndentStr, IndentStr, Label, IndentStr, Bind, Placeholder, Rows, IndentStr])
    ;
        format(atom(Code), '~w<textarea v-model="~w" placeholder="~w" rows="~w" style="width: 100%; padding: 10px;"></textarea>~n',
               [IndentStr, Bind, Placeholder, Rows])
    ).

% Checkbox component
generate_component(checkbox, Options, Indent, _GenOpts, Code) :-
    get_option(bind, Options, Bind, ''),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<label style="display: flex; align-items: center; gap: 8px;"><input type="checkbox" v-model="~w"> ~w</label>~n',
           [IndentStr, Bind, Label]).

% Select component
generate_component(select, Options, Indent, _GenOpts, Code) :-
    get_option(bind, Options, Bind, ''),
    get_option(options, Options, Opts, []),
    get_option(placeholder, Options, Placeholder, 'Select...'),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),

    (is_binding(Opts) ->
        binding_to_vue(Opts, VueOpts),
        format(atom(OptionsCode), '<option v-for="opt in ~w" :key="opt.value" :value="opt.value">{{ opt.label }}</option>', [VueOpts])
    ;
        generate_static_options(Opts, OptionsCode)
    ),

    (Label \= '' ->
        format(atom(Code),
'~w<div class="form-group">
~w  <label>~w</label>
~w  <select v-model="~w" style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
~w    <option value="" disabled>~w</option>
~w    ~w
~w  </select>
~w</div>~n',
               [IndentStr, IndentStr, Label, IndentStr, Bind, IndentStr, Placeholder, IndentStr, OptionsCode, IndentStr, IndentStr])
    ;
        format(atom(Code), '~w<select v-model="~w" style="width: 100%; padding: 10px;"><option value="" disabled>~w</option>~w</select>~n',
               [IndentStr, Bind, Placeholder, OptionsCode])
    ).

% Switch component
generate_component(switch, Options, Indent, _GenOpts, Code) :-
    get_option(bind, Options, Bind, ''),
    get_option(label, Options, Label, ''),

    indent_string(Indent, IndentStr),
    format(atom(Code),
'~w<label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
~w  <input type="checkbox" v-model="~w" style="display: none;">
~w  <span :style="{ width: \'40px\', height: \'20px\', background: ~w ? \'#4ade80\' : \'#ccc\', borderRadius: \'10px\', position: \'relative\', transition: \'0.3s\' }">
~w    <span :style="{ position: \'absolute\', top: \'2px\', left: ~w ? \'22px\' : \'2px\', width: \'16px\', height: \'16px\', background: \'#fff\', borderRadius: \'50%\', transition: \'0.3s\' }"></span>
~w  </span>
~w  ~w
~w</label>~n',
           [IndentStr, IndentStr, Bind, IndentStr, Bind, IndentStr, Bind, IndentStr, IndentStr, Label, IndentStr]).

% Tabs component
generate_component(tabs, Options, Indent, _GenOpts, Code) :-
    get_option(items, Options, Items, []),
    get_option(active, Options, Active, ''),
    get_option(on_change, Options, OnChange, ''),

    indent_string(Indent, IndentStr),

    (is_binding(Items) ->
        binding_to_vue(Items, VueItems),
        (OnChange \= '' ->
            format(atom(TabCode), '<button v-for="item in ~w" :key="item" @click="~w = item; ~w(item)" :class="{ active: ~w === item }" class="tab">{{ item }}</button>',
                   [VueItems, Active, OnChange, Active])
        ;
            format(atom(TabCode), '<button v-for="item in ~w" :key="item" @click="~w = item" :class="{ active: ~w === item }" class="tab">{{ item }}</button>',
                   [VueItems, Active, Active])
        )
    ;
        generate_static_tabs(Items, Active, OnChange, TabCode)
    ),

    format(atom(Code), '~w<div class="tabs" style="display: flex; gap: 5px;">~w</div>~n', [IndentStr, TabCode]).

% Spinner component
generate_component(spinner, Options, Indent, _GenOpts, Code) :-
    get_option(size, Options, Size, 24),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="width: ~wpx; height: ~wpx; border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>~n',
           [IndentStr, Size, Size]).

% Progress component
generate_component(progress, Options, Indent, _GenOpts, Code) :-
    get_option(value, Options, Value, 0),
    get_option(max, Options, Max, 100),

    indent_string(Indent, IndentStr),
    (is_binding(Value) ->
        binding_to_vue(Value, VueValue),
        format(atom(Code), '~w<div style="width: 100%; height: 8px; background: #eee; border-radius: 4px; overflow: hidden;"><div :style="{ width: (~w / ~w * 100) + \'%\', height: \'100%\', background: \'#3498db\' }"></div></div>~n',
               [IndentStr, VueValue, Max])
    ;
        Percent is Value / Max * 100,
        format(atom(Code), '~w<div style="width: 100%; height: 8px; background: #eee; border-radius: 4px; overflow: hidden;"><div style="width: ~w%; height: 100%; background: #3498db;"></div></div>~n',
               [IndentStr, Percent])
    ).

% Badge component
generate_component(badge, Options, Indent, _GenOpts, Code) :-
    get_option(content, Options, Content, ''),
    get_option(variant, Options, Variant, info),

    variant_badge_style(Variant, BadgeStyle),
    indent_string(Indent, IndentStr),

    (is_binding(Content) ->
        binding_to_vue(Content, VueContent),
        format(atom(Code), '~w<span style="~w">{{ ~w }}</span>~n', [IndentStr, BadgeStyle, VueContent])
    ;
        format(atom(Code), '~w<span style="~w">~w</span>~n', [IndentStr, BadgeStyle, Content])
    ).

% Divider component
generate_component(divider, Options, Indent, _GenOpts, Code) :-
    get_option(margin, Options, Margin, 16),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<hr style="border: none; border-top: 1px solid #eee; margin: ~wpx 0;">~n', [IndentStr, Margin]).

% Spacer component
generate_component(spacer, Options, Indent, _GenOpts, Code) :-
    get_option(size, Options, Size, 16),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="height: ~wpx;"></div>~n', [IndentStr, Size]).

% Image component
generate_component(image, Options, Indent, _GenOpts, Code) :-
    get_option(src, Options, Src, ''),
    get_option(alt, Options, Alt, ''),
    get_option(width, Options, Width, ''),
    get_option(height, Options, Height, ''),

    indent_string(Indent, IndentStr),
    build_dimension_style(Width, Height, DimStyle),

    (is_binding(Src) ->
        binding_to_vue(Src, VueSrc),
        format(atom(Code), '~w<img :src="~w" alt="~w"~w>~n', [IndentStr, VueSrc, Alt, DimStyle])
    ;
        format(atom(Code), '~w<img src="~w" alt="~w"~w>~n', [IndentStr, Src, Alt, DimStyle])
    ).

% Avatar component
generate_component(avatar, Options, Indent, _GenOpts, Code) :-
    get_option(src, Options, Src, ''),
    get_option(name, Options, Name, ''),
    get_option(size, Options, Size, 40),

    indent_string(Indent, IndentStr),
    (   Src \= ''
    ->  format(atom(Code), '~w<img src="~w" alt="~w" style="width: ~wpx; height: ~wpx; border-radius: 50%; object-fit: cover;">~n',
               [IndentStr, Src, Name, Size, Size])
    ;   format(atom(Code), '~w<div style="width: ~wpx; height: ~wpx; border-radius: 50%; background: #3498db; display: flex; justify-content: center; align-items: center; color: #fff; font-weight: bold; overflow: hidden;">~w</div>~n',
               [IndentStr, Size, Size, Name])
    ).

% Icon component
generate_component(icon, Options, Indent, _GenOpts, Code) :-
    get_option(name, Options, Name, ''),
    get_option(size, Options, Size, 24),

    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<span style="font-size: ~wpx;">~w</span>~n', [IndentStr, Size, Name]).

% Alert component
generate_component(alert, Options, Indent, _GenOpts, Code) :-
    get_option(message, Options, Message, ''),
    get_option(variant, Options, Variant, info),

    variant_alert_style(Variant, AlertStyle),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<div style="~w">~w</div>~n', [IndentStr, AlertStyle, Message]).

% Default component fallback
generate_component(Type, Options, Indent, _GenOpts, Code) :-
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<!-- Component: ~w ~w -->~n', [IndentStr, Type, Options]).

% ============================================================================
% CONDITIONAL GENERATION
% ============================================================================

generate_conditional(Condition, Content, Indent, GenOpts, Code) :-
    condition_to_vue(Condition, VueCond),
    NextIndent is Indent + 1,
    generate_node(Content, NextIndent, GenOpts, ContentCode),
    indent_string(Indent, IndentStr),
    format(atom(Code), '~w<template v-if="~w">~n~w~w</template>~n',
           [IndentStr, VueCond, ContentCode, IndentStr]).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

%! generate_children(+Children, +Indent, +Options, -Code) is det
generate_children([], _Indent, _Options, '').
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

% Style helpers
direction_style(row, 'flex-direction: row').
direction_style(column, 'flex-direction: column').

spacing_style(0, '') :- !.
spacing_style(N, Style) :- format(atom(Style), 'gap: ~wpx', [N]).

gap_style(0, '') :- !.
gap_style(N, Style) :- format(atom(Style), 'gap: ~wpx', [N]).

padding_style(0, '') :- !.
padding_style(N, Style) :- format(atom(Style), 'padding: ~wpx', [N]).

align_style(start, 'align-items: flex-start').
align_style(center, 'align-items: center').
align_style(end, 'align-items: flex-end').
align_style(stretch, 'align-items: stretch').
align_style(_, '').

justify_style(start, 'justify-content: flex-start').
justify_style(center, 'justify-content: center').
justify_style(end, 'justify-content: flex-end').
justify_style(between, 'justify-content: space-between').
justify_style(around, 'justify-content: space-around').
justify_style(evenly, 'justify-content: space-evenly').
justify_style(_, '').

wrap_style(true, 'flex-wrap: wrap').
wrap_style(false, '').
wrap_style(_, '').

shadow_style(0, '') :- !.
shadow_style(1, 'box-shadow: 0 1px 3px rgba(0,0,0,0.12)').
shadow_style(2, 'box-shadow: 0 3px 6px rgba(0,0,0,0.15)').
shadow_style(3, 'box-shadow: 0 10px 20px rgba(0,0,0,0.19)').
shadow_style(_, 'box-shadow: 0 3px 6px rgba(0,0,0,0.15)').

position_styles(Options, Styles) :-
    findall(S, (
        member(Opt, Options),
        Opt =.. [Key, Val],
        member(Key, [top, right, bottom, left, z_index]),
        format(atom(S), '~w: ~w', [Key, Val])
    ), StyleList),
    atomic_list_concat(StyleList, '; ', Styles).

% Variant styles
variant_button_style(primary, 'padding: 10px 20px; background: #e94560; border: none; color: #fff; cursor: pointer; border-radius: 5px; font-weight: bold').
variant_button_style(secondary, 'padding: 10px 20px; background: #16213e; border: none; color: #fff; cursor: pointer; border-radius: 5px').
variant_button_style(danger, 'padding: 10px 20px; background: #dc3545; border: none; color: #fff; cursor: pointer; border-radius: 5px').
variant_button_style(_, 'padding: 10px 20px; background: #e94560; border: none; color: #fff; cursor: pointer; border-radius: 5px').

variant_badge_style(info, 'padding: 2px 8px; background: #3498db; color: #fff; border-radius: 3px; font-size: 12px').
variant_badge_style(success, 'padding: 2px 8px; background: #4ade80; color: #000; border-radius: 3px; font-size: 12px').
variant_badge_style(warning, 'padding: 2px 8px; background: #fbbf24; color: #000; border-radius: 3px; font-size: 12px').
variant_badge_style(error, 'padding: 2px 8px; background: #ff6b6b; color: #fff; border-radius: 3px; font-size: 12px').
variant_badge_style(_, 'padding: 2px 8px; background: #ccc; border-radius: 3px; font-size: 12px').

variant_alert_style(info, 'padding: 12px 16px; background: #d1ecf1; color: #0c5460; border-radius: 5px').
variant_alert_style(success, 'padding: 12px 16px; background: #d4edda; color: #155724; border-radius: 5px').
variant_alert_style(warning, 'padding: 12px 16px; background: #fff3cd; color: #856404; border-radius: 5px').
variant_alert_style(error, 'padding: 12px 16px; background: #f8d7da; color: #721c24; border-radius: 5px').
variant_alert_style(_, 'padding: 12px 16px; background: #eee; border-radius: 5px').

% Binding detection
is_binding(Term) :- atom(Term), atom_codes(Term, [C|_]), C \= 34, C \= 39.  % Not starting with quote
is_binding(bind(X)) :- atom(X).

binding_to_vue(bind(X), X) :- !.
binding_to_vue(X, X) :- atom(X).

% Condition conversion
condition_to_vue(Cond, Cond) :- atom(Cond), !.
condition_to_vue(not(C), VueCond) :-
    condition_to_vue(C, VC),
    format(atom(VueCond), '!~w', [VC]).
condition_to_vue(and(A, B), VueCond) :-
    condition_to_vue(A, VA),
    condition_to_vue(B, VB),
    format(atom(VueCond), '(~w && ~w)', [VA, VB]).
condition_to_vue(or(A, B), VueCond) :-
    condition_to_vue(A, VA),
    condition_to_vue(B, VB),
    format(atom(VueCond), '(~w || ~w)', [VA, VB]).
condition_to_vue([H|T], VueCond) :-
    condition_to_vue(H, VH),
    condition_to_vue(T, VT),
    format(atom(VueCond), '(~w && ~w)', [VH, VT]).
condition_to_vue([], 'true').

items_to_vue(Items, Items) :- atom(Items), !.
items_to_vue(Items, VueItems) :- is_list(Items), term_to_atom(Items, VueItems).

% Build style attribute
build_style_attr([], '', '') :- !.
build_style_attr(Pairs, BaseStyle, StyleAttr) :-
    findall(S, (member(K-V, Pairs), V \= '', format(atom(S), '~w: ~w', [K, V])), Styles),
    (BaseStyle \= '' -> append(Styles, [BaseStyle], AllStyles) ; AllStyles = Styles),
    (AllStyles \= [] ->
        atomic_list_concat(AllStyles, '; ', StyleStr),
        format(atom(StyleAttr), ' style="~w"', [StyleStr])
    ;
        StyleAttr = ''
    ).

build_dimension_style('', '', '') :- !.
build_dimension_style(W, '', Style) :- W \= '', format(atom(Style), ' style="width: ~w"', [W]).
build_dimension_style('', H, Style) :- H \= '', format(atom(Style), ' style="height: ~w"', [H]).
build_dimension_style(W, H, Style) :- W \= '', H \= '', format(atom(Style), ' style="width: ~w; height: ~w"', [W, H]).

generate_static_options([], '').
generate_static_options([Opt|Rest], Code) :-
    (Opt = opt(Value, Label) ->
        format(atom(OptCode), '<option value="~w">~w</option>', [Value, Label])
    ;
        format(atom(OptCode), '<option value="~w">~w</option>', [Opt, Opt])
    ),
    generate_static_options(Rest, RestCode),
    atom_concat(OptCode, RestCode, Code).

generate_static_tabs([], _Active, _OnChange, '').
generate_static_tabs([Tab|Rest], Active, OnChange, Code) :-
    (OnChange \= '' ->
        format(atom(TabCode), '<button @click="~w = \'~w\'; ~w(\'~w\')" :class="{ active: ~w === \'~w\' }" class="tab">~w</button>',
               [Active, Tab, OnChange, Tab, Active, Tab, Tab])
    ;
        format(atom(TabCode), '<button @click="~w = \'~w\'" :class="{ active: ~w === \'~w\' }" class="tab">~w</button>',
               [Active, Tab, Active, Tab, Tab])
    ),
    generate_static_tabs(Rest, Active, OnChange, RestCode),
    atom_concat(TabCode, RestCode, Code).

% ============================================================================
% SFC GENERATION
% ============================================================================

%! generate_vue_sfc(+UISpec, +Options, -SFC) is det
%  Generate a complete Vue Single File Component.
generate_vue_sfc(UISpec, Options, SFC) :-
    generate_vue_template(UISpec, Options, Template),
    generate_vue_script(UISpec, Options, Script),
    generate_vue_styles(UISpec, Options, Styles),
    format(atom(SFC), '<template>~n~w</template>~n~n<script setup>~n~w</script>~n~n<style scoped>~n~w</style>~n',
           [Template, Script, Styles]).

%! generate_vue_script(+UISpec, +Options, -Script) is det
%  Generate Vue script section.
generate_vue_script(_UISpec, _Options, Script) :-
    Script = 'import { ref, reactive } from \'vue\';\n\n// TODO: Extract reactive state from UI spec\n'.

%! generate_vue_styles(+UISpec, +Options, -Styles) is det
%  Generate Vue styles section.
generate_vue_styles(_UISpec, _Options, Styles) :-
    Styles = '.tab { padding: 10px 20px; background: #16213e; border: none; color: #94a3b8; cursor: pointer; border-radius: 5px 5px 0 0; }\n.tab.active { background: #0f3460; color: #e94560; }\n'.

% ============================================================================
% TESTING
% ============================================================================

test_vue_generator :-
    format('~n=== Vue Generator Tests ===~n~n'),

    % Test 1: Simple text
    format('Test 1: Simple text component...~n'),
    generate_vue_template(component(text, [content("Hello World")]), [], T1),
    format('  Output: ~w~n', [T1]),

    % Test 2: Button
    format('~nTest 2: Button component...~n'),
    generate_vue_template(component(button, [label("Click Me"), on_click(handleClick)]), [], T2),
    format('  Output: ~w~n', [T2]),

    % Test 3: Stack layout
    format('~nTest 3: Stack layout...~n'),
    UI3 = layout(stack, [spacing(16)], [
        component(heading, [level(1), content("Title")]),
        component(text, [content("Description")])
    ]),
    generate_vue_template(UI3, [], T3),
    format('  Output:~n~w~n', [T3]),

    % Test 4: Form with inputs
    format('~nTest 4: Form with inputs...~n'),
    UI4 = layout(stack, [spacing(12)], [
        component(text_input, [label("Email"), type(email), bind(email), placeholder("Enter email")]),
        component(text_input, [label("Password"), type(password), bind(password)]),
        component(button, [label("Submit"), on_click(submit), variant(primary)])
    ]),
    generate_vue_template(UI4, [], T4),
    format('  Output:~n~w~n', [T4]),

    % Test 5: Conditional
    format('~nTest 5: Conditional rendering...~n'),
    UI5 = when(isLoggedIn, component(text, [content("Welcome!")])),
    generate_vue_template(UI5, [], T5),
    format('  Output: ~w~n', [T5]),

    % Test 6: Foreach
    format('~nTest 6: Foreach iteration...~n'),
    UI6 = foreach(items, item, component(text, [content(item)])),
    generate_vue_template(UI6, [], T6),
    format('  Output: ~w~n', [T6]),

    % Test 7: Nested containers
    format('~nTest 7: Nested containers...~n'),
    UI7 = container(panel, [padding(20), background('#1a1a2e')], [
        layout(stack, [spacing(8)], [
            component(heading, [level(2), content("Panel Title")]),
            component(text, [content("Panel content goes here")])
        ])
    ]),
    generate_vue_template(UI7, [], T7),
    format('  Output:~n~w~n', [T7]),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('Vue Generator module loaded~n')
), now).
