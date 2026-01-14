% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% i18n.pl - Internationalization and Localization Module
%
% Provides translation key management, interpolation, and pluralization
% for generating localized UI code across all target frameworks.
%
% Key features:
%   - Translation key definition: t('key')
%   - Variable interpolation: {{name}} syntax
%   - Pluralization: zero/one/other forms
%   - Multi-locale support
%   - Target-specific code generation
%
% Usage:
%   ?- define_translations(en, ['greeting' - "Hello, {{name}}!"]).
%   ?- resolve_translation('greeting', en, [name('World')], Text).
%   ?- extract_translation_keys(Spec, Keys).

:- module(i18n, [
    % Translation definition
    define_translations/2,      % +Locale, +Translations
    add_translation/3,          % +Locale, +Key, +Value
    get_translation/3,          % +Locale, +Key, -Value
    clear_translations/0,       % Clear all translations
    clear_translations/1,       % +Locale - Clear locale translations

    % Translation keys (term constructors)
    t/2,                        % +Key, -Term
    t/3,                        % +Key, +Vars, -Term
    t_plural/3,                 % +Key, +Count, -Term

    % Resolution
    resolve_translation/4,      % +Key, +Locale, +Vars, -Text
    resolve_plural/4,           % +Key, +Count, +Locale, -Text

    % Spec transformation
    localize_spec/3,            % +Spec, +Locale, -LocalizedSpec
    extract_translation_keys/2, % +Spec, -Keys

    % Code generation
    generate_i18n_setup/3,      % +Target, +Locales, -Code
    generate_translation_file/3, % +Target, +Locale, -Content
    generate_translation_json/2, % +Locale, -JSON

    % Validation
    find_missing_translations/3, % +Spec, +Locale, -MissingKeys
    validate_translations/2,     % +Locale, -Errors
    list_locales/1,             % -Locales

    % Testing
    test_i18n/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_translation/3.  % stored_translation(Locale, Key, Value)
:- dynamic default_locale/1.      % default_locale(Locale)

% Default locale
:- initialization((
    (   default_locale(_)
    ->  true
    ;   assertz(default_locale(en))
    )
), now).

% ============================================================================
% TRANSLATION DEFINITION
% ============================================================================

%% define_translations(+Locale, +Translations)
%
%  Define multiple translations for a locale.
%
%  Translations: [Key - Value, ...]
%
define_translations(Locale, Translations) :-
    atom(Locale),
    is_list(Translations),
    forall(member(Key - Value, Translations),
           add_translation(Locale, Key, Value)).

%% add_translation(+Locale, +Key, +Value)
%
%  Add a single translation.
%
add_translation(Locale, Key, Value) :-
    atom(Locale),
    (atom(Key) ; string(Key)),
    (atom(Value) ; string(Value)),
    % Convert key to atom if string
    (string(Key) -> atom_string(KeyAtom, Key) ; KeyAtom = Key),
    % Convert value to string for consistency
    (atom(Value) -> atom_string(ValueStr, Value) ; ValueStr = Value),
    retractall(stored_translation(Locale, KeyAtom, _)),
    assertz(stored_translation(Locale, KeyAtom, ValueStr)).

%% get_translation(+Locale, +Key, -Value)
%
%  Get a translation value.
%
get_translation(Locale, Key, Value) :-
    (string(Key) -> atom_string(KeyAtom, Key) ; KeyAtom = Key),
    stored_translation(Locale, KeyAtom, Value).

%% clear_translations/0
%
%  Clear all translations.
%
clear_translations :-
    retractall(stored_translation(_, _, _)).

%% clear_translations(+Locale)
%
%  Clear translations for a specific locale.
%
clear_translations(Locale) :-
    retractall(stored_translation(Locale, _, _)).

%% list_locales(-Locales)
%
%  Get list of all defined locales.
%
list_locales(Locales) :-
    findall(L, stored_translation(L, _, _), AllLocales),
    sort(AllLocales, Locales).

% ============================================================================
% TRANSLATION KEY CONSTRUCTORS
% ============================================================================

%% t(+Key, -Term)
%
%  Create a translation key term.
%
t(Key, t(Key)).

%% t(+Key, +Vars, -Term)
%
%  Create a translation key term with variables.
%
t(Key, Vars, t(Key, Vars)).

%% t_plural(+Key, +Count, -Term)
%
%  Create a plural translation key term.
%
t_plural(Key, Count, t_plural(Key, Count)).

% ============================================================================
% TRANSLATION RESOLUTION
% ============================================================================

%% resolve_translation(+Key, +Locale, +Vars, -Text)
%
%  Resolve a translation key to text with variable interpolation.
%
resolve_translation(Key, Locale, Vars, Text) :-
    (string(Key) -> atom_string(KeyAtom, Key) ; KeyAtom = Key),
    (   get_translation(Locale, KeyAtom, Template)
    ->  interpolate_vars(Template, Vars, Text)
    ;   % Fallback to default locale
        default_locale(DefaultLocale),
        DefaultLocale \= Locale,
        get_translation(DefaultLocale, KeyAtom, Template)
    ->  interpolate_vars(Template, Vars, Text)
    ;   % Return key as fallback
        atom_string(KeyAtom, Text)
    ).

%% interpolate_vars(+Template, +Vars, -Result)
%
%  Replace {{var}} placeholders with values.
%
interpolate_vars(Template, [], Template) :- !.
interpolate_vars(Template, Vars, Result) :-
    string(Template),
    foldl(replace_var, Vars, Template, Result).

replace_var(VarSpec, In, Out) :-
    VarSpec =.. [Name, Value],
    atom_string(Name, NameStr),
    format(string(Placeholder), "{{~w}}", [NameStr]),
    (   atom(Value)
    ->  atom_string(Value, ValueStr)
    ;   number(Value)
    ->  number_string(Value, ValueStr)
    ;   ValueStr = Value
    ),
    replace_all_occurrences(In, Placeholder, ValueStr, Out).

replace_all_occurrences(In, Search, Replace, Out) :-
    (   sub_string(In, Before, Len, After, Search),
        string_length(Search, Len)
    ->  sub_string(In, 0, Before, _, Prefix),
        AfterStart is Before + Len,
        sub_string(In, AfterStart, After, 0, Suffix),
        string_concat(Prefix, Replace, Temp),
        replace_all_occurrences(Suffix, Search, Replace, RestOut),
        string_concat(Temp, RestOut, Out)
    ;   Out = In
    ).

%% resolve_plural(+Key, +Count, +Locale, -Text)
%
%  Resolve a plural translation.
%
%  Looks for keys: key.zero, key.one, key.other
%
resolve_plural(Key, Count, Locale, Text) :-
    (string(Key) -> atom_string(KeyAtom, Key) ; KeyAtom = Key),
    plural_form(Count, Form),
    atom_concat(KeyAtom, '.', KeyPrefix),
    atom_concat(KeyPrefix, Form, PluralKey),
    (   get_translation(Locale, PluralKey, Template)
    ->  interpolate_vars(Template, [count(Count)], Text)
    ;   % Fallback to .other
        atom_concat(KeyPrefix, other, OtherKey),
        get_translation(Locale, OtherKey, OtherTemplate)
    ->  interpolate_vars(OtherTemplate, [count(Count)], Text)
    ;   % Fallback to base key
        get_translation(Locale, KeyAtom, BaseTemplate)
    ->  interpolate_vars(BaseTemplate, [count(Count)], Text)
    ;   format(string(Text), "~w (~d)", [KeyAtom, Count])
    ).

plural_form(0, zero).
plural_form(1, one).
plural_form(N, other) :- N \= 0, N \= 1.

% ============================================================================
% SPEC TRANSFORMATION
% ============================================================================

%% extract_translation_keys(+Spec, -Keys)
%
%  Extract all t() keys from a specification.
%
extract_translation_keys(Spec, Keys) :-
    extract_keys_acc(Spec, [], KeyList),
    sort(KeyList, Keys).

extract_keys_acc(t(Key), Acc, [Key|Acc]) :- !.
extract_keys_acc(t(Key, _), Acc, [Key|Acc]) :- !.
extract_keys_acc(t_plural(Key, _), Acc, [Key|Acc]) :- !.

extract_keys_acc(Spec, Acc, Result) :-
    compound(Spec),
    Spec =.. [_|Args],
    foldl(extract_keys_from_arg, Args, Acc, Result),
    !.

extract_keys_acc(_, Acc, Acc).

extract_keys_from_arg(Arg, Acc, Result) :-
    is_list(Arg),
    !,
    foldl(extract_keys_acc, Arg, Acc, Result).

extract_keys_from_arg(Arg, Acc, Result) :-
    extract_keys_acc(Arg, Acc, Result).

%% localize_spec(+Spec, +Locale, -LocalizedSpec)
%
%  Replace t() terms with resolved translations.
%
localize_spec(t(Key), Locale, Text) :-
    !,
    resolve_translation(Key, Locale, [], Text).

localize_spec(t(Key, Vars), Locale, Text) :-
    !,
    resolve_translation(Key, Locale, Vars, Text).

localize_spec(t_plural(Key, Count), Locale, Text) :-
    !,
    resolve_plural(Key, Count, Locale, Text).

localize_spec(Spec, Locale, Localized) :-
    compound(Spec),
    Spec =.. [Functor|Args],
    maplist(localize_spec_wrapper(Locale), Args, LocalizedArgs),
    Localized =.. [Functor|LocalizedArgs],
    !.

localize_spec(Spec, _, Spec).

localize_spec_wrapper(Locale, Arg, Result) :-
    is_list(Arg),
    !,
    maplist(localize_spec_wrapper(Locale), Arg, Result).

localize_spec_wrapper(Locale, Arg, Result) :-
    localize_spec(Arg, Locale, Result).

% ============================================================================
% CODE GENERATION
% ============================================================================

%% generate_i18n_setup(+Target, +Locales, -Code)
%
%  Generate i18n setup code for a target framework.
%
generate_i18n_setup(react_native, Locales, Code) :-
    generate_react_i18n_setup(Locales, Code).

generate_i18n_setup(vue, Locales, Code) :-
    generate_vue_i18n_setup(Locales, Code).

generate_i18n_setup(flutter, Locales, Code) :-
    generate_flutter_i18n_setup(Locales, Code).

generate_i18n_setup(swiftui, Locales, Code) :-
    generate_swiftui_i18n_setup(Locales, Code).

%% generate_react_i18n_setup(+Locales, -Code)
generate_react_i18n_setup(Locales, Code) :-
    maplist(locale_import_line, Locales, ImportLines),
    atomic_list_concat(ImportLines, '\n', ImportsStr),
    maplist(locale_resource_entry, Locales, ResourceEntries),
    atomic_list_concat(ResourceEntries, ',\n    ', ResourcesStr),
    format(string(Code),
"import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

~w

i18n.use(initReactI18next).init({
  resources: {
    ~w
  },
  lng: 'en',
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false
  }
});

export default i18n;", [ImportsStr, ResourcesStr]).

locale_import_line(Locale, Line) :-
    format(string(Line), "import ~w from './locales/~w.json';", [Locale, Locale]).

locale_resource_entry(Locale, Entry) :-
    format(string(Entry), "~w: { translation: ~w }", [Locale, Locale]).

%% generate_vue_i18n_setup(+Locales, -Code)
generate_vue_i18n_setup(Locales, Code) :-
    maplist(locale_import_line, Locales, ImportLines),
    atomic_list_concat(ImportLines, '\n', ImportsStr),
    maplist(vue_message_entry, Locales, MessageEntries),
    atomic_list_concat(MessageEntries, ', ', MessagesStr),
    format(string(Code),
"import { createI18n } from 'vue-i18n';

~w

export const i18n = createI18n({
  locale: 'en',
  fallbackLocale: 'en',
  messages: { ~w }
});", [ImportsStr, MessagesStr]).

vue_message_entry(Locale, Entry) :-
    format(string(Entry), "~w", [Locale]).

%% generate_flutter_i18n_setup(+Locales, -Code)
generate_flutter_i18n_setup(Locales, Code) :-
    maplist(flutter_locale_entry, Locales, LocaleEntries),
    atomic_list_concat(LocaleEntries, ',\n    ', LocalesStr),
    format(string(Code),
"import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

class AppLocalizations {
  final Locale locale;

  AppLocalizations(this.locale);

  static AppLocalizations of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations)!;
  }

  static const LocalizationsDelegate<AppLocalizations> delegate = _AppLocalizationsDelegate();

  static const List<Locale> supportedLocales = [
    ~w
  ];
}

class _AppLocalizationsDelegate extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  bool isSupported(Locale locale) => ['en', 'es'].contains(locale.languageCode);

  @override
  Future<AppLocalizations> load(Locale locale) async {
    return AppLocalizations(locale);
  }

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}", [LocalesStr]).

flutter_locale_entry(Locale, Entry) :-
    format(string(Entry), "Locale('~w')", [Locale]).

%% generate_swiftui_i18n_setup(+Locales, -Code)
generate_swiftui_i18n_setup(_Locales, Code) :-
    Code = "import SwiftUI

// SwiftUI uses native localization with Localizable.strings files
// Translation keys are used with LocalizedStringKey(\"key\")

extension Text {
    init(localized key: String) {
        self.init(LocalizedStringKey(key))
    }
}".

%% generate_translation_file(+Target, +Locale, -Content)
%
%  Generate a translation file for a specific target and locale.
%
generate_translation_file(react_native, Locale, Content) :-
    generate_translation_json(Locale, Content).

generate_translation_file(vue, Locale, Content) :-
    generate_translation_json(Locale, Content).

generate_translation_file(flutter, Locale, Content) :-
    generate_flutter_arb(Locale, Content).

generate_translation_file(swiftui, Locale, Content) :-
    generate_strings_file(Locale, Content).

%% generate_translation_json(+Locale, -JSON)
%
%  Generate JSON translation file content.
%
generate_translation_json(Locale, JSON) :-
    findall(Entry, (
        stored_translation(Locale, Key, Value),
        format(string(Entry), "  \"~w\": \"~w\"", [Key, Value])
    ), Entries),
    atomic_list_concat(Entries, ',\n', EntriesStr),
    format(string(JSON), "{\n~w\n}", [EntriesStr]).

%% generate_flutter_arb(+Locale, -ARB)
generate_flutter_arb(Locale, ARB) :-
    findall(Entry, (
        stored_translation(Locale, Key, Value),
        arb_key(Key, ArbKey),
        format(string(Entry), "  \"~w\": \"~w\"", [ArbKey, Value])
    ), Entries),
    atomic_list_concat(Entries, ',\n', EntriesStr),
    format(string(ARB), "{\n  \"@@locale\": \"~w\",\n~w\n}", [Locale, EntriesStr]).

arb_key(Key, ArbKey) :-
    atom_string(Key, KeyStr),
    replace_all_occurrences(KeyStr, ".", "_", ArbKeyStr),
    atom_string(ArbKey, ArbKeyStr).

%% generate_strings_file(+Locale, -Content)
generate_strings_file(Locale, Content) :-
    findall(Entry, (
        stored_translation(Locale, Key, Value),
        format(string(Entry), "\"~w\" = \"~w\";", [Key, Value])
    ), Entries),
    atomic_list_concat(Entries, '\n', Content).

% ============================================================================
% VALIDATION
% ============================================================================

%% find_missing_translations(+Spec, +Locale, -MissingKeys)
%
%  Find translation keys used in spec but not defined for locale.
%
find_missing_translations(Spec, Locale, MissingKeys) :-
    extract_translation_keys(Spec, UsedKeys),
    findall(Key, (
        member(Key, UsedKeys),
        \+ get_translation(Locale, Key, _)
    ), MissingKeys).

%% validate_translations(+Locale, -Errors)
%
%  Validate translations for a locale.
%
validate_translations(Locale, Errors) :-
    findall(Error, validate_translation_error(Locale, Error), Errors).

validate_translation_error(Locale, error(empty_value, Key)) :-
    stored_translation(Locale, Key, Value),
    string_length(Value, 0).

validate_translation_error(Locale, error(unbalanced_braces, Key)) :-
    stored_translation(Locale, Key, Value),
    \+ balanced_interpolation(Value).

balanced_interpolation(Value) :-
    string(Value),
    findall(_, sub_string(Value, _, _, _, "{{"), Opens),
    findall(_, sub_string(Value, _, _, _, "}}"), Closes),
    length(Opens, N),
    length(Closes, N).

% ============================================================================
% TESTING
% ============================================================================

test_i18n :-
    format('~n=== i18n Tests ===~n~n'),

    % Setup
    clear_translations,

    % Test 1: Define translations
    format('Test 1: Define translations...~n'),
    define_translations(en, [
        'greeting' - "Hello!",
        'welcome' - "Welcome, {{name}}!",
        'items.zero' - "No items",
        'items.one' - "1 item",
        'items.other' - "{{count}} items"
    ]),
    (   get_translation(en, 'greeting', "Hello!")
    ->  format('  PASS: Translations defined~n')
    ;   format('  FAIL: Translations not defined~n')
    ),

    % Test 2: Simple resolution
    format('~nTest 2: Simple resolution...~n'),
    (   resolve_translation('greeting', en, [], Text1),
        Text1 = "Hello!"
    ->  format('  PASS: Resolved to: ~w~n', [Text1])
    ;   format('  FAIL: Resolution broken~n')
    ),

    % Test 3: Interpolation
    format('~nTest 3: Interpolation...~n'),
    (   resolve_translation('welcome', en, [name('World')], Text2),
        Text2 = "Welcome, World!"
    ->  format('  PASS: Interpolated: ~w~n', [Text2])
    ;   format('  FAIL: Interpolation broken~n')
    ),

    % Test 4: Pluralization - zero
    format('~nTest 4: Pluralization (zero)...~n'),
    (   resolve_plural('items', 0, en, Text3),
        Text3 = "No items"
    ->  format('  PASS: Plural zero: ~w~n', [Text3])
    ;   format('  FAIL: Plural zero broken~n')
    ),

    % Test 5: Pluralization - one
    format('~nTest 5: Pluralization (one)...~n'),
    (   resolve_plural('items', 1, en, Text4),
        Text4 = "1 item"
    ->  format('  PASS: Plural one: ~w~n', [Text4])
    ;   format('  FAIL: Plural one broken~n')
    ),

    % Test 6: Pluralization - other
    format('~nTest 6: Pluralization (other)...~n'),
    (   resolve_plural('items', 5, en, Text5),
        Text5 = "5 items"
    ->  format('  PASS: Plural other: ~w~n', [Text5])
    ;   format('  FAIL: Plural other broken~n')
    ),

    % Test 7: Key extraction
    format('~nTest 7: Key extraction...~n'),
    TestSpec = nav([title(t('nav.home')), icon(t('nav.icon'))]),
    (   extract_translation_keys(TestSpec, Keys),
        member('nav.home', Keys),
        member('nav.icon', Keys)
    ->  format('  PASS: Extracted keys: ~w~n', [Keys])
    ;   format('  FAIL: Key extraction broken~n')
    ),

    % Test 8: JSON generation
    format('~nTest 8: JSON generation...~n'),
    (   generate_translation_json(en, JSON),
        sub_string(JSON, _, _, _, "greeting"),
        sub_string(JSON, _, _, _, "Hello!")
    ->  format('  PASS: JSON generated~n')
    ;   format('  FAIL: JSON generation broken~n')
    ),

    % Test 9: Missing translation detection
    format('~nTest 9: Missing translation detection...~n'),
    MissingSpec = screen([title(t('undefined.key'))]),
    (   find_missing_translations(MissingSpec, en, Missing),
        member('undefined.key', Missing)
    ->  format('  PASS: Found missing keys: ~w~n', [Missing])
    ;   format('  FAIL: Missing detection broken~n')
    ),

    % Test 10: i18n setup generation
    format('~nTest 10: i18n setup generation...~n'),
    (   generate_i18n_setup(react_native, [en, es], SetupCode),
        sub_string(SetupCode, _, _, _, "i18next"),
        sub_string(SetupCode, _, _, _, "initReactI18next")
    ->  format('  PASS: Setup code generated~n')
    ;   format('  FAIL: Setup generation broken~n')
    ),

    % Cleanup
    clear_translations,

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('i18n module loaded~n', [])
), now).
