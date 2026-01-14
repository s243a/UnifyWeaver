% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_i18n.pl - plunit tests for i18n module
%
% Tests translation definition, interpolation, pluralization,
% and code generation for all target frameworks.
%
% Run with: swipl -g "run_tests" -t halt test_i18n.pl

:- module(test_i18n, []).

:- use_module(library(plunit)).
:- use_module('i18n').

% ============================================================================
% Setup/Cleanup
% ============================================================================

setup_test_translations :-
    i18n:clear_translations,
    i18n:define_translations(en, [
        'greeting' - "Hello!",
        'welcome' - "Welcome, {{name}}!",
        'farewell' - "Goodbye, {{name}}. See you {{when}}!",
        'items.zero' - "No items",
        'items.one' - "1 item",
        'items.other' - "{{count}} items",
        'nav.home' - "Home",
        'nav.cart' - "Cart",
        'auth.login' - "Login",
        'auth.error.required' - "This field is required"
    ]),
    i18n:define_translations(es, [
        'greeting' - "Hola!",
        'welcome' - "Bienvenido, {{name}}!",
        'items.zero' - "Sin articulos",
        'items.one' - "1 articulo",
        'items.other' - "{{count}} articulos",
        'nav.home' - "Inicio",
        'nav.cart' - "Carrito"
    ]).

cleanup_test_translations :-
    i18n:clear_translations.

% ============================================================================
% Tests: Translation Definition
% ============================================================================

:- begin_tests(translation_definition, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(define_simple_translation) :-
    i18n:get_translation(en, 'greeting', Value),
    Value = "Hello!".

test(define_multiple_translations) :-
    i18n:get_translation(en, 'greeting', _),
    i18n:get_translation(en, 'welcome', _),
    i18n:get_translation(en, 'nav.home', _).

test(define_multiple_locales) :-
    i18n:get_translation(en, 'greeting', "Hello!"),
    i18n:get_translation(es, 'greeting', "Hola!").

test(list_locales) :-
    i18n:list_locales(Locales),
    member(en, Locales),
    member(es, Locales).

test(overwrite_translation) :-
    i18n:add_translation(en, 'test.overwrite', "Original"),
    i18n:add_translation(en, 'test.overwrite', "Updated"),
    i18n:get_translation(en, 'test.overwrite', "Updated").

:- end_tests(translation_definition).

% ============================================================================
% Tests: Translation Resolution
% ============================================================================

:- begin_tests(translation_resolution, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(resolve_simple) :-
    i18n:resolve_translation('greeting', en, [], Text),
    Text = "Hello!".

test(resolve_different_locale) :-
    i18n:resolve_translation('greeting', es, [], Text),
    Text = "Hola!".

test(resolve_missing_returns_key) :-
    i18n:resolve_translation('nonexistent.key', en, [], Text),
    Text = "nonexistent.key".

test(resolve_dotted_key) :-
    i18n:resolve_translation('nav.home', en, [], Text),
    Text = "Home".

:- end_tests(translation_resolution).

% ============================================================================
% Tests: Interpolation
% ============================================================================

:- begin_tests(interpolation, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(interpolate_single_var) :-
    i18n:resolve_translation('welcome', en, [name('World')], Text),
    Text = "Welcome, World!".

test(interpolate_multiple_vars) :-
    i18n:resolve_translation('farewell', en, [name('John'), when('tomorrow')], Text),
    Text = "Goodbye, John. See you tomorrow!".

test(interpolate_number) :-
    i18n:add_translation(en, 'test.number', "Count: {{n}}"),
    i18n:resolve_translation('test.number', en, [n(42)], Text),
    Text = "Count: 42".

test(interpolate_missing_var_unchanged) :-
    i18n:resolve_translation('welcome', en, [], Text),
    sub_string(Text, _, _, _, "{{name}}").

:- end_tests(interpolation).

% ============================================================================
% Tests: Pluralization
% ============================================================================

:- begin_tests(pluralization, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(plural_zero) :-
    i18n:resolve_plural('items', 0, en, Text),
    Text = "No items".

test(plural_one) :-
    i18n:resolve_plural('items', 1, en, Text),
    Text = "1 item".

test(plural_other_small) :-
    i18n:resolve_plural('items', 5, en, Text),
    Text = "5 items".

test(plural_other_large) :-
    i18n:resolve_plural('items', 100, en, Text),
    Text = "100 items".

test(plural_spanish) :-
    i18n:resolve_plural('items', 5, es, Text),
    Text = "5 articulos".

test(plural_fallback_to_other) :-
    i18n:add_translation(en, 'only.other', "{{count}} things"),
    i18n:resolve_plural('only', 7, en, Text),
    Text = "7 things".

:- end_tests(pluralization).

% ============================================================================
% Tests: Key Extraction
% ============================================================================

:- begin_tests(key_extraction, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(extract_single_key) :-
    Spec = title(t('nav.home')),
    i18n:extract_translation_keys(Spec, Keys),
    Keys = ['nav.home'].

test(extract_multiple_keys) :-
    Spec = nav([t('nav.home'), t('nav.cart')]),
    i18n:extract_translation_keys(Spec, Keys),
    length(Keys, 2),
    member('nav.home', Keys),
    member('nav.cart', Keys).

test(extract_nested_keys) :-
    Spec = screen([
        header(t('greeting')),
        body([text(t('welcome'))])
    ]),
    i18n:extract_translation_keys(Spec, Keys),
    member('greeting', Keys),
    member('welcome', Keys).

test(extract_no_duplicates) :-
    Spec = tabs([t('nav.home'), t('nav.home'), t('nav.cart')]),
    i18n:extract_translation_keys(Spec, Keys),
    length(Keys, 2).

test(extract_empty_spec) :-
    Spec = simple(value),
    i18n:extract_translation_keys(Spec, Keys),
    Keys = [].

:- end_tests(key_extraction).

% ============================================================================
% Tests: Spec Localization
% ============================================================================

:- begin_tests(spec_localization, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(localize_simple_spec) :-
    Spec = title(t('greeting')),
    i18n:localize_spec(Spec, en, Localized),
    Localized = title("Hello!").

test(localize_nested_spec) :-
    Spec = nav([screen(home, t('nav.home')), screen(cart, t('nav.cart'))]),
    i18n:localize_spec(Spec, en, Localized),
    Localized = nav([screen(home, "Home"), screen(cart, "Cart")]).

test(localize_different_locale) :-
    Spec = title(t('nav.home')),
    i18n:localize_spec(Spec, es, Localized),
    Localized = title("Inicio").

test(localize_with_interpolation) :-
    Spec = message(t('welcome', [name('User')])),
    i18n:localize_spec(Spec, en, Localized),
    Localized = message("Welcome, User!").

:- end_tests(spec_localization).

% ============================================================================
% Tests: Missing Translation Detection
% ============================================================================

:- begin_tests(missing_detection, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(find_missing_single) :-
    Spec = title(t('undefined.key')),
    i18n:find_missing_translations(Spec, en, Missing),
    Missing = ['undefined.key'].

test(find_missing_multiple) :-
    Spec = screen([t('undefined1'), t('undefined2')]),
    i18n:find_missing_translations(Spec, en, Missing),
    length(Missing, 2).

test(find_missing_partial) :-
    Spec = screen([t('nav.home'), t('undefined')]),
    i18n:find_missing_translations(Spec, en, Missing),
    Missing = ['undefined'].

test(find_missing_none) :-
    Spec = screen([t('nav.home'), t('nav.cart')]),
    i18n:find_missing_translations(Spec, en, Missing),
    Missing = [].

test(find_missing_different_locale) :-
    Spec = title(t('auth.login')),
    i18n:find_missing_translations(Spec, es, Missing),
    Missing = ['auth.login'].

:- end_tests(missing_detection).

% ============================================================================
% Tests: JSON Generation
% ============================================================================

:- begin_tests(json_generation, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(json_has_braces) :-
    i18n:generate_translation_json(en, JSON),
    sub_string(JSON, 0, _, _, "{"),
    sub_string(JSON, _, _, 0, "}").

test(json_has_keys) :-
    i18n:generate_translation_json(en, JSON),
    sub_string(JSON, _, _, _, "\"greeting\""),
    sub_string(JSON, _, _, _, "\"nav.home\"").

test(json_has_values) :-
    i18n:generate_translation_json(en, JSON),
    sub_string(JSON, _, _, _, "Hello!"),
    sub_string(JSON, _, _, _, "Home").

test(json_different_locale) :-
    i18n:generate_translation_json(es, JSON),
    sub_string(JSON, _, _, _, "Hola!"),
    sub_string(JSON, _, _, _, "Inicio").

:- end_tests(json_generation).

% ============================================================================
% Tests: i18n Setup Generation
% ============================================================================

:- begin_tests(setup_generation, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(react_native_setup) :-
    i18n:generate_i18n_setup(react_native, [en, es], Code),
    sub_string(Code, _, _, _, "i18next"),
    sub_string(Code, _, _, _, "initReactI18next"),
    sub_string(Code, _, _, _, "en.json"),
    sub_string(Code, _, _, _, "es.json").

test(vue_setup) :-
    i18n:generate_i18n_setup(vue, [en, es], Code),
    sub_string(Code, _, _, _, "vue-i18n"),
    sub_string(Code, _, _, _, "createI18n").

test(flutter_setup) :-
    i18n:generate_i18n_setup(flutter, [en, es], Code),
    sub_string(Code, _, _, _, "AppLocalizations"),
    sub_string(Code, _, _, _, "LocalizationsDelegate").

test(swiftui_setup) :-
    i18n:generate_i18n_setup(swiftui, [en, es], Code),
    sub_string(Code, _, _, _, "SwiftUI"),
    sub_string(Code, _, _, _, "LocalizedStringKey").

:- end_tests(setup_generation).

% ============================================================================
% Tests: Validation
% ============================================================================

:- begin_tests(validation, [setup(setup_test_translations), cleanup(cleanup_test_translations)]).

test(validate_empty_value) :-
    i18n:add_translation(en, 'test.empty', ""),
    i18n:validate_translations(en, Errors),
    member(error(empty_value, 'test.empty'), Errors).

test(validate_unbalanced_braces) :-
    i18n:add_translation(en, 'test.unbalanced', "Hello {{name"),
    i18n:validate_translations(en, Errors),
    member(error(unbalanced_braces, 'test.unbalanced'), Errors).

test(validate_good_translations) :-
    i18n:clear_translations,
    i18n:add_translation(en, 'test.good', "Valid translation"),
    i18n:validate_translations(en, Errors),
    Errors = [].

:- end_tests(validation).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
