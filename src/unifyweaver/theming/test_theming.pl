% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_theming.pl - plunit tests for theming module
%
% Run with: swipl -g "run_tests" -t halt test_theming.pl

:- module(test_theming, []).

:- use_module(library(plunit)).
:- use_module('theming').

% ============================================================================
% Setup/Cleanup
% ============================================================================

setup_test_theme :-
    clear_themes,
    define_theme(test_app, [
        colors([
            primary-'#007AFF',
            secondary-'#5856D6',
            success-'#34C759',
            warning-'#FF9500',
            error-'#FF3B30',
            background-'#FFFFFF',
            surface-'#F2F2F7',
            text-'#000000'
        ]),
        typography([
            family-'Inter',
            sizeXs-12,
            sizeSm-14,
            sizeMd-16,
            sizeLg-20,
            sizeXl-24,
            weightNormal-400,
            weightBold-700
        ]),
        spacing([
            xs-4,
            sm-8,
            md-16,
            lg-24,
            xl-32
        ]),
        borders([
            radiusSm-4,
            radiusMd-8,
            radiusLg-16,
            radiusFull-9999
        ])
    ]),
    define_variant(test_app, dark, [
        colors([
            background-'#1C1C1E',
            surface-'#2C2C2E',
            text-'#FFFFFF'
        ])
    ]).

cleanup_test_theme :-
    clear_themes.

% ============================================================================
% Tests: Theme Definition
% ============================================================================

:- begin_tests(theme_definition, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(define_and_get_theme) :-
    get_theme(test_app, Theme),
    is_list(Theme).

test(theme_has_colors) :-
    get_theme(test_app, Theme),
    member(colors(_), Theme).

test(theme_has_typography) :-
    get_theme(test_app, Theme),
    member(typography(_), Theme).

test(theme_has_spacing) :-
    get_theme(test_app, Theme),
    member(spacing(_), Theme).

test(list_themes_includes_test) :-
    list_themes(Names),
    member(test_app, Names).

test(redefine_theme) :-
    define_theme(temp_theme, [colors([a-'#111'])]),
    define_theme(temp_theme, [colors([b-'#222'])]),
    get_theme(temp_theme, Theme),
    member(colors(Colors), Theme),
    member(b-'#222', Colors),
    \+ member(a-'#111', Colors).

:- end_tests(theme_definition).

% ============================================================================
% Tests: Color Values
% ============================================================================

:- begin_tests(color_values, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(primary_color) :-
    get_theme(test_app, Theme),
    member(colors(Colors), Theme),
    member(primary-'#007AFF', Colors).

test(multiple_colors) :-
    get_theme(test_app, Theme),
    member(colors(Colors), Theme),
    member(success-'#34C759', Colors),
    member(error-'#FF3B30', Colors).

test(background_color) :-
    get_theme(test_app, Theme),
    member(colors(Colors), Theme),
    member(background-'#FFFFFF', Colors).

:- end_tests(color_values).

% ============================================================================
% Tests: Typography Values
% ============================================================================

:- begin_tests(typography_values, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(font_family) :-
    get_theme(test_app, Theme),
    member(typography(Typo), Theme),
    member(family-'Inter', Typo).

test(font_sizes) :-
    get_theme(test_app, Theme),
    member(typography(Typo), Theme),
    member(sizeMd-16, Typo),
    member(sizeLg-20, Typo).

test(font_weights) :-
    get_theme(test_app, Theme),
    member(typography(Typo), Theme),
    member(weightNormal-400, Typo),
    member(weightBold-700, Typo).

:- end_tests(typography_values).

% ============================================================================
% Tests: Spacing Values
% ============================================================================

:- begin_tests(spacing_values, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(spacing_scale) :-
    get_theme(test_app, Theme),
    member(spacing(Spaces), Theme),
    member(xs-4, Spaces),
    member(md-16, Spaces),
    member(xl-32, Spaces).

test(spacing_order) :-
    get_theme(test_app, Theme),
    member(spacing(Spaces), Theme),
    member(sm-SmVal, Spaces),
    member(lg-LgVal, Spaces),
    SmVal < LgVal.

:- end_tests(spacing_values).

% ============================================================================
% Tests: Theme Variants
% ============================================================================

:- begin_tests(theme_variants, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(dark_variant_exists) :-
    get_variant(test_app, dark, _).

test(dark_has_overridden_background) :-
    get_variant(test_app, dark, Theme),
    member(colors(Colors), Theme),
    member(background-'#1C1C1E', Colors).

test(dark_has_overridden_text) :-
    get_variant(test_app, dark, Theme),
    member(colors(Colors), Theme),
    member(text-'#FFFFFF', Colors).

:- end_tests(theme_variants).

% ============================================================================
% Tests: Token Resolution
% ============================================================================

:- begin_tests(token_resolution, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(resolve_color_token) :-
    get_theme(test_app, Theme),
    resolve_token(token(colors, primary), Theme, '#007AFF').

test(resolve_spacing_token) :-
    get_theme(test_app, Theme),
    resolve_token(token(spacing, md), Theme, 16).

test(resolve_typography_token) :-
    get_theme(test_app, Theme),
    resolve_token(token(typography, sizeLg), Theme, 20).

:- end_tests(token_resolution).

% ============================================================================
% Tests: React Native Generation
% ============================================================================

:- begin_tests(react_native_theme, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(rn_has_export) :-
    generate_theme_code(test_app, react_native, Code),
    sub_string(Code, _, _, _, "export const theme").

test(rn_has_colors_object) :-
    generate_theme_code(test_app, react_native, Code),
    sub_string(Code, _, _, _, "colors:").

test(rn_has_primary_color) :-
    generate_theme_code(test_app, react_native, Code),
    sub_string(Code, _, _, _, "#007AFF").

test(rn_has_typography) :-
    generate_theme_code(test_app, react_native, Code),
    sub_string(Code, _, _, _, "typography:").

test(rn_has_spacing) :-
    generate_theme_code(test_app, react_native, Code),
    sub_string(Code, _, _, _, "spacing:").

:- end_tests(react_native_theme).

% ============================================================================
% Tests: Vue Generation
% ============================================================================

:- begin_tests(vue_theme, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(vue_has_root) :-
    generate_theme_code(test_app, vue, Code),
    sub_string(Code, _, _, _, ":root").

test(vue_has_color_vars) :-
    generate_theme_code(test_app, vue, Code),
    sub_string(Code, _, _, _, "--color-primary").

test(vue_has_spacing_vars) :-
    generate_theme_code(test_app, vue, Code),
    sub_string(Code, _, _, _, "--spacing-md").

test(vue_has_font_vars) :-
    generate_theme_code(test_app, vue, Code),
    sub_string(Code, _, _, _, "--font-").

:- end_tests(vue_theme).

% ============================================================================
% Tests: Flutter Generation
% ============================================================================

:- begin_tests(flutter_theme, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(flutter_has_class) :-
    generate_theme_code(test_app, flutter, Code),
    sub_string(Code, _, _, _, "class AppTheme").

test(flutter_has_color_const) :-
    generate_theme_code(test_app, flutter, Code),
    sub_string(Code, _, _, _, "static const Color").

test(flutter_has_hex_color) :-
    generate_theme_code(test_app, flutter, Code),
    sub_string(Code, _, _, _, "Color(0xFF").

:- end_tests(flutter_theme).

% ============================================================================
% Tests: SwiftUI Generation
% ============================================================================

:- begin_tests(swiftui_theme, [setup(setup_test_theme), cleanup(cleanup_test_theme)]).

test(swift_has_import) :-
    generate_theme_code(test_app, swiftui, Code),
    sub_string(Code, _, _, _, "import SwiftUI").

test(swift_has_struct) :-
    generate_theme_code(test_app, swiftui, Code),
    sub_string(Code, _, _, _, "struct Theme").

test(swift_has_colors_struct) :-
    generate_theme_code(test_app, swiftui, Code),
    sub_string(Code, _, _, _, "struct Colors").

test(swift_has_color_hex) :-
    generate_theme_code(test_app, swiftui, Code),
    sub_string(Code, _, _, _, "Color(hex:").

:- end_tests(swiftui_theme).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
