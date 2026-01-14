% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% theming.pl - Cross-platform theme system
%
% Provides theme definitions (colors, typography, spacing) that compile
% across React Native, Vue, Flutter, and SwiftUI targets.
%
% Usage:
%   use_module('src/unifyweaver/theming/theming').
%   define_theme(my_app, [colors([primary('#007AFF')]), ...]),
%   generate_theme_code(my_app, react_native, Code).

:- module(theming, [
    % Theme definition
    define_theme/2,
    get_theme/2,
    clear_themes/0,
    list_themes/1,

    % Theme components
    colors/1,
    typography/1,
    spacing/1,
    borders/1,
    shadows/1,

    % Color utilities
    color/2,
    color_variant/3,
    opacity/2,

    % Typography utilities
    font_family/2,
    font_size/2,
    font_weight/2,
    line_height/2,

    % Spacing utilities
    space/2,
    padding/2,
    margin/2,

    % Theme variants
    define_variant/3,
    get_variant/3,
    apply_variant/3,

    % Token resolution
    resolve_token/3,
    resolve_all_tokens/3,

    % Code generation
    generate_theme_code/3,
    generate_react_native_theme/2,
    generate_vue_theme/2,
    generate_flutter_theme/2,
    generate_swiftui_theme/2,

    % Testing
    test_theming/0
]).

:- use_module(library(lists)).

% ============================================================================
% Dynamic Theme Storage
% ============================================================================

:- dynamic theme/2.
:- dynamic theme_variant/3.

%! clear_themes is det
%  Clear all defined themes.
clear_themes :-
    retractall(theme(_, _)),
    retractall(theme_variant(_, _, _)).

%! define_theme(+Name, +Definition) is det
%  Define a new theme.
define_theme(Name, Definition) :-
    retractall(theme(Name, _)),
    assertz(theme(Name, Definition)).

%! get_theme(+Name, -Definition) is semidet
%  Get a theme definition.
get_theme(Name, Definition) :-
    theme(Name, Definition).

%! list_themes(-Names) is det
%  List all defined theme names.
list_themes(Names) :-
    findall(Name, theme(Name, _), Names).

% ============================================================================
% Theme Component Constructors
% ============================================================================

%! colors(+ColorList) is det
%  Define theme colors.
colors(ColorList) :- is_list(ColorList).

%! typography(+TypographyList) is det
%  Define theme typography.
typography(TypographyList) :- is_list(TypographyList).

%! spacing(+SpacingList) is det
%  Define theme spacing scale.
spacing(SpacingList) :- is_list(SpacingList).

%! borders(+BorderList) is det
%  Define theme borders.
borders(BorderList) :- is_list(BorderList).

%! shadows(+ShadowList) is det
%  Define theme shadows.
shadows(ShadowList) :- is_list(ShadowList).

% ============================================================================
% Color Utilities
% ============================================================================

%! color(+Name, +Value) is det
%  Define a named color.
color(Name, Value) :-
    atom(Name),
    (atom(Value) ; is_list(Value)).

%! color_variant(+Base, +Variant, +Value) is det
%  Define a color variant (e.g., primary_light).
color_variant(Base, Variant, Value) :-
    atom(Base),
    atom(Variant),
    atom(Value).

%! opacity(+Color, +Alpha) is det
%  Apply opacity to a color.
opacity(Color, Alpha) :-
    atom(Color),
    number(Alpha),
    Alpha >= 0,
    Alpha =< 1.

% ============================================================================
% Typography Utilities
% ============================================================================

%! font_family(+Name, +Family) is det
font_family(Name, Family) :- atom(Name), atom(Family).

%! font_size(+Name, +Size) is det
font_size(Name, Size) :- atom(Name), number(Size).

%! font_weight(+Name, +Weight) is det
font_weight(Name, Weight) :- atom(Name), (atom(Weight) ; number(Weight)).

%! line_height(+Name, +Height) is det
line_height(Name, Height) :- atom(Name), number(Height).

% ============================================================================
% Spacing Utilities
% ============================================================================

%! space(+Name, +Value) is det
space(Name, Value) :- atom(Name), number(Value).

%! padding(+Name, +Value) is det
padding(Name, Value) :- atom(Name), (number(Value) ; is_list(Value)).

%! margin(+Name, +Value) is det
margin(Name, Value) :- atom(Name), (number(Value) ; is_list(Value)).

% ============================================================================
% Theme Variants (Light/Dark)
% ============================================================================

%! define_variant(+ThemeName, +VariantName, +Overrides) is det
%  Define a theme variant with overrides.
define_variant(ThemeName, VariantName, Overrides) :-
    retractall(theme_variant(ThemeName, VariantName, _)),
    assertz(theme_variant(ThemeName, VariantName, Overrides)).

%! get_variant(+ThemeName, +VariantName, -Definition) is semidet
%  Get a theme variant.
get_variant(ThemeName, VariantName, Definition) :-
    theme(ThemeName, BaseTheme),
    theme_variant(ThemeName, VariantName, Overrides),
    merge_theme(BaseTheme, Overrides, Definition).

%! apply_variant(+Theme, +VariantName, -Result) is det
%  Apply a variant to a theme.
apply_variant(Theme, VariantName, Result) :-
    (   theme_variant(_, VariantName, Overrides)
    ->  merge_theme(Theme, Overrides, Result)
    ;   Result = Theme
    ).

merge_theme(Base, Overrides, Merged) :-
    findall(Component, (
        (member(Component, Overrides) ; member(Component, Base)),
        \+ (Component =.. [Name|_], member(Override, Overrides), Override =.. [Name|_], Component \= Override)
    ), MergedList),
    sort(MergedList, Merged).

% ============================================================================
% Token Resolution
% ============================================================================

%! resolve_token(+Token, +Theme, -Value) is semidet
%  Resolve a theme token to its value.
resolve_token(token(Category, Name), Theme, Value) :-
    member(CategoryDef, Theme),
    CategoryDef =.. [Category, Items],
    member(Item, Items),
    Item =.. [Name, Value].

resolve_token(token(Category, Name), Theme, Value) :-
    member(CategoryDef, Theme),
    CategoryDef =.. [Category, Items],
    member(Name-Value, Items).

%! resolve_all_tokens(+Spec, +Theme, -Resolved) is det
%  Resolve all tokens in a spec.
resolve_all_tokens(token(Cat, Name), Theme, Value) :-
    !,
    (   resolve_token(token(Cat, Name), Theme, Value)
    ->  true
    ;   Value = token(Cat, Name)  % Keep unresolved
    ).
resolve_all_tokens(Spec, Theme, Resolved) :-
    is_list(Spec),
    !,
    maplist({Theme}/[S, R]>>resolve_all_tokens(S, Theme, R), Spec, Resolved).
resolve_all_tokens(Spec, Theme, Resolved) :-
    compound(Spec),
    !,
    Spec =.. [Functor|Args],
    maplist({Theme}/[A, R]>>resolve_all_tokens(A, Theme, R), Args, ResolvedArgs),
    Resolved =.. [Functor|ResolvedArgs].
resolve_all_tokens(Spec, _, Spec).

% ============================================================================
% Code Generation - Main Entry Point
% ============================================================================

%! generate_theme_code(+ThemeName, +Target, -Code) is det
%  Generate target-specific theme code.
generate_theme_code(ThemeName, Target, Code) :-
    get_theme(ThemeName, Theme),
    (   Target = react_native
    ->  generate_react_native_theme(Theme, Code)
    ;   Target = vue
    ->  generate_vue_theme(Theme, Code)
    ;   Target = flutter
    ->  generate_flutter_theme(Theme, Code)
    ;   Target = swiftui
    ->  generate_swiftui_theme(Theme, Code)
    ;   Code = ""
    ).

% ============================================================================
% React Native Theme Generation
% ============================================================================

%! generate_react_native_theme(+Theme, -Code) is det
generate_react_native_theme(Theme, Code) :-
    generate_rn_colors(Theme, ColorsCode),
    generate_rn_typography(Theme, TypographyCode),
    generate_rn_spacing(Theme, SpacingCode),
    format(atom(Code), 'export const theme = {\n  colors: ~w,\n  typography: ~w,\n  spacing: ~w,\n};\n',
           [ColorsCode, TypographyCode, SpacingCode]).

generate_rn_colors(Theme, Code) :-
    (   member(colors(Colors), Theme)
    ->  findall(Entry, (
            member(C, Colors),
            format_rn_color(C, Entry)
        ), Entries),
        atomic_list_concat(Entries, ', ', EntriesStr),
        format(atom(Code), '{ ~w }', [EntriesStr])
    ;   Code = '{}'
    ).

format_rn_color(Name-Value, Entry) :-
    !,
    format(atom(Entry), '~w: \'~w\'', [Name, Value]).
format_rn_color(Color, Entry) :-
    Color =.. [Name, Value],
    format(atom(Entry), '~w: \'~w\'', [Name, Value]).

generate_rn_typography(Theme, Code) :-
    (   member(typography(Typo), Theme)
    ->  findall(Entry, (
            member(T, Typo),
            format_rn_typo(T, Entry)
        ), Entries),
        atomic_list_concat(Entries, ', ', EntriesStr),
        format(atom(Code), '{ ~w }', [EntriesStr])
    ;   Code = '{}'
    ).

format_rn_typo(Name-Value, Entry) :-
    !,
    format(atom(Entry), '~w: ~w', [Name, Value]).
format_rn_typo(Typo, Entry) :-
    Typo =.. [Name, Value],
    (   number(Value)
    ->  format(atom(Entry), '~w: ~w', [Name, Value])
    ;   format(atom(Entry), '~w: \'~w\'', [Name, Value])
    ).

generate_rn_spacing(Theme, Code) :-
    (   member(spacing(Spaces), Theme)
    ->  findall(Entry, (
            member(S, Spaces),
            format_rn_spacing(S, Entry)
        ), Entries),
        atomic_list_concat(Entries, ', ', EntriesStr),
        format(atom(Code), '{ ~w }', [EntriesStr])
    ;   Code = '{}'
    ).

format_rn_spacing(Name-Value, Entry) :-
    !,
    format(atom(Entry), '~w: ~w', [Name, Value]).
format_rn_spacing(Space, Entry) :-
    Space =.. [Name, Value],
    format(atom(Entry), '~w: ~w', [Name, Value]).

% ============================================================================
% Vue Theme Generation
% ============================================================================

%! generate_vue_theme(+Theme, -Code) is det
generate_vue_theme(Theme, Code) :-
    generate_vue_css_vars(Theme, CSSVars),
    format(atom(Code), ':root {\n~w}\n', [CSSVars]).

generate_vue_css_vars(Theme, Code) :-
    findall(Var, (
        member(colors(Colors), Theme),
        member(C, Colors),
        format_css_color_var(C, Var)
    ), ColorVars),
    findall(Var, (
        member(spacing(Spaces), Theme),
        member(S, Spaces),
        format_css_spacing_var(S, Var)
    ), SpacingVars),
    findall(Var, (
        member(typography(Typo), Theme),
        member(T, Typo),
        format_css_typo_var(T, Var)
    ), TypoVars),
    append([ColorVars, SpacingVars, TypoVars], AllVars),
    atomic_list_concat(AllVars, '', Code).

format_css_color_var(Name-Value, Var) :-
    !,
    format(atom(Var), '  --color-~w: ~w;\n', [Name, Value]).
format_css_color_var(Color, Var) :-
    Color =.. [Name, Value],
    format(atom(Var), '  --color-~w: ~w;\n', [Name, Value]).

format_css_spacing_var(Name-Value, Var) :-
    !,
    format(atom(Var), '  --spacing-~w: ~wpx;\n', [Name, Value]).
format_css_spacing_var(Space, Var) :-
    Space =.. [Name, Value],
    format(atom(Var), '  --spacing-~w: ~wpx;\n', [Name, Value]).

format_css_typo_var(Name-Value, Var) :-
    !,
    format(atom(Var), '  --font-~w: ~w;\n', [Name, Value]).
format_css_typo_var(Typo, Var) :-
    Typo =.. [Name, Value],
    (   number(Value)
    ->  format(atom(Var), '  --font-~w: ~wpx;\n', [Name, Value])
    ;   format(atom(Var), '  --font-~w: ~w;\n', [Name, Value])
    ).

% ============================================================================
% Flutter Theme Generation
% ============================================================================

%! generate_flutter_theme(+Theme, -Code) is det
generate_flutter_theme(Theme, Code) :-
    generate_flutter_colors(Theme, ColorsCode),
    generate_flutter_text_theme(Theme, TextCode),
    format(atom(Code), 'class AppTheme {\n~w\n~w}\n', [ColorsCode, TextCode]).

generate_flutter_colors(Theme, Code) :-
    (   member(colors(Colors), Theme)
    ->  findall(Line, (
            member(C, Colors),
            format_flutter_color(C, Line)
        ), Lines),
        atomic_list_concat(Lines, '', Code)
    ;   Code = ''
    ).

format_flutter_color(Name-Value, Line) :-
    !,
    hex_to_flutter(Value, FlutterColor),
    format(atom(Line), '  static const Color ~w = ~w;\n', [Name, FlutterColor]).
format_flutter_color(Color, Line) :-
    Color =.. [Name, Value],
    hex_to_flutter(Value, FlutterColor),
    format(atom(Line), '  static const Color ~w = ~w;\n', [Name, FlutterColor]).

hex_to_flutter(Hex, FlutterColor) :-
    (   atom_concat('#', HexVal, Hex)
    ->  format(atom(FlutterColor), 'Color(0xFF~w)', [HexVal])
    ;   format(atom(FlutterColor), 'Color(0xFF~w)', [Hex])
    ).

generate_flutter_text_theme(Theme, Code) :-
    (   member(typography(Typo), Theme)
    ->  findall(Line, (
            member(T, Typo),
            format_flutter_typo(T, Line)
        ), Lines),
        atomic_list_concat(Lines, '', Code)
    ;   Code = ''
    ).

format_flutter_typo(Name-Value, Line) :-
    !,
    format(atom(Line), '  static const double fontSize~w = ~w;\n', [Name, Value]).
format_flutter_typo(Typo, Line) :-
    Typo =.. [Name, Value],
    (   number(Value)
    ->  format(atom(Line), '  static const double fontSize~w = ~w;\n', [Name, Value])
    ;   format(atom(Line), '  static const String font~w = \'~w\';\n', [Name, Value])
    ).

% ============================================================================
% SwiftUI Theme Generation
% ============================================================================

%! generate_swiftui_theme(+Theme, -Code) is det
generate_swiftui_theme(Theme, Code) :-
    generate_swift_colors(Theme, ColorsCode),
    generate_swift_fonts(Theme, FontsCode),
    format(atom(Code), 'import SwiftUI\n\nstruct Theme {\n  struct Colors {\n~w  }\n\n  struct Fonts {\n~w  }\n}\n',
           [ColorsCode, FontsCode]).

generate_swift_colors(Theme, Code) :-
    (   member(colors(Colors), Theme)
    ->  findall(Line, (
            member(C, Colors),
            format_swift_color(C, Line)
        ), Lines),
        atomic_list_concat(Lines, '', Code)
    ;   Code = ''
    ).

format_swift_color(Name-Value, Line) :-
    !,
    format(atom(Line), '    static let ~w = Color(hex: \"~w\")\n', [Name, Value]).
format_swift_color(Color, Line) :-
    Color =.. [Name, Value],
    format(atom(Line), '    static let ~w = Color(hex: \"~w\")\n', [Name, Value]).

generate_swift_fonts(Theme, Code) :-
    (   member(typography(Typo), Theme)
    ->  findall(Line, (
            member(T, Typo),
            format_swift_font(T, Line)
        ), Lines),
        atomic_list_concat(Lines, '', Code)
    ;   Code = ''
    ).

format_swift_font(Name-Value, Line) :-
    !,
    (   number(Value)
    ->  format(atom(Line), '    static let ~w: CGFloat = ~w\n', [Name, Value])
    ;   format(atom(Line), '    static let ~w = \"~w\"\n', [Name, Value])
    ).
format_swift_font(Typo, Line) :-
    Typo =.. [Name, Value],
    (   number(Value)
    ->  format(atom(Line), '    static let ~w: CGFloat = ~w\n', [Name, Value])
    ;   format(atom(Line), '    static let ~w = \"~w\"\n', [Name, Value])
    ).

% ============================================================================
% Testing
% ============================================================================

%! test_theming is det
%  Run inline tests.
test_theming :-
    format('Running theming tests...~n'),
    clear_themes,

    % Test 1: Define theme
    define_theme(test_theme, [
        colors([primary-'#007AFF', secondary-'#5856D6', background-'#FFFFFF']),
        typography([family-'Inter', sizeBase-16, sizeLarge-20]),
        spacing([xs-4, sm-8, md-16, lg-24])
    ]),
    get_theme(test_theme, _),
    format('  Test 1 passed: define theme~n'),

    % Test 2: List themes
    list_themes(Themes),
    member(test_theme, Themes),
    format('  Test 2 passed: list themes~n'),

    % Test 3: React Native code generation
    generate_theme_code(test_theme, react_native, RNCode),
    sub_string(RNCode, _, _, _, "export const theme"),
    sub_string(RNCode, _, _, _, "colors:"),
    sub_string(RNCode, _, _, _, "#007AFF"),
    format('  Test 3 passed: React Native generation~n'),

    % Test 4: Vue code generation
    generate_theme_code(test_theme, vue, VueCode),
    sub_string(VueCode, _, _, _, ":root"),
    sub_string(VueCode, _, _, _, "--color-primary"),
    format('  Test 4 passed: Vue generation~n'),

    % Test 5: Flutter code generation
    generate_theme_code(test_theme, flutter, FlutterCode),
    sub_string(FlutterCode, _, _, _, "class AppTheme"),
    sub_string(FlutterCode, _, _, _, "Color(0xFF"),
    format('  Test 5 passed: Flutter generation~n'),

    % Test 6: SwiftUI code generation
    generate_theme_code(test_theme, swiftui, SwiftCode),
    sub_string(SwiftCode, _, _, _, "struct Theme"),
    sub_string(SwiftCode, _, _, _, "Color(hex:"),
    format('  Test 6 passed: SwiftUI generation~n'),

    % Test 7: Theme variants
    define_variant(test_theme, dark, [
        colors([background-'#1C1C1E', text-'#FFFFFF'])
    ]),
    get_variant(test_theme, dark, DarkTheme),
    member(colors(DarkColors), DarkTheme),
    member(background-'#1C1C1E', DarkColors),
    format('  Test 7 passed: theme variants~n'),

    % Test 8: Token resolution
    get_theme(test_theme, Theme8),
    resolve_token(token(colors, primary), Theme8, '#007AFF'),
    format('  Test 8 passed: token resolution~n'),

    % Test 9: Spacing values
    get_theme(test_theme, Theme9),
    member(spacing(Spaces), Theme9),
    member(md-16, Spaces),
    format('  Test 9 passed: spacing values~n'),

    % Test 10: Typography values
    get_theme(test_theme, Theme10),
    member(typography(Typo), Theme10),
    member(sizeBase-16, Typo),
    format('  Test 10 passed: typography values~n'),

    clear_themes,
    format('All 10 theming tests passed!~n'),
    !.

:- initialization(test_theming, main).
