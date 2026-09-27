% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% annotated_js_target.pl - Annotated JavaScript (JSDoc) Target
%
% A *variant* of the TypeScript target (typescript_target.pl) that emits
% plain JavaScript annotated with JSDoc type comments. Follows the same
% inheritance pattern clojurescript_target uses for clojure_target: this
% module `use_module`s its base target and only overrides the type-annotation
% emission (TS inline types → JSDoc; .ts → .js; interface/generics →
% @typedef/@param/@returns).
%
%   annotated_js_target : typescript_target
%       ::  clojurescript_target : clojure_target
%
% The shipped artifact is the exact .js file you read, edit, and debug.
% It runs unmodified on Node/browser and type-checks under
% `npx tsc --checkJs --noEmit --allowJs` with no build step and no runtime
% dependency. tsc is a dev-only checker, never a compiler.
%
% Recursion patterns are inherited from the TypeScript target
% (tail_recursion, linear_recursion, list_fold, transitive_closure) and
% only post-processed by ts_to_annotated_js/2.
%
% Example:
%   ?- compile_recursion(factorial/2, [pattern(linear_recursion)], Code).
%   ?- write_annotated_js_module(Code, 'factorial.js').

:- module(annotated_js_target, [
    % Standard interface
    target_info/1,                  % -Info
    compile_predicate/3,            % +Pred/Arity, +Options, -Code
    compile_facts/3,                % +Pred, +Arity, -Code
    compile_recursion/3,            % +Pred/Arity, +Options, -Code
    compile_module/3,               % +Predicates, +Options, -Code
    write_annotated_js_module/2,    % +Code, +Filename
    init_annotated_js_target/0,

    % Binding system (delegates to TypeScript)
    clear_binding_imports/0,
    collect_binding_import/1,
    get_collected_imports/1,

    % TS → JSDoc rewrite (mirrors clojurescript_interop_rewrite/2)
    ts_to_annotated_js/2            % +TSCode, -JSCode
]).

:- use_module(library(lists)).
:- use_module(library(option)).

% Inherit the TypeScript target. Override only type-annotation emission.
:- use_module(typescript_target, except([
    target_info/1,
    compile_predicate/3,
    compile_facts/3,
    compile_recursion/3,
    compile_module/3,
    clear_binding_imports/0,
    collect_binding_import/1,
    get_collected_imports/1
])).

%% ============================================
%% TARGET INFO
%% ============================================

target_info(info{
    name: "AnnotatedJS",
    family: javascript,
    file_extension: ".js",
    runtime: auto,
    features: [jsdoc, tsc_checked, modules, async],
    recursion_patterns: [tail_recursion, linear_recursion, list_fold, transitive_closure],
    compile_command: "npx tsc --checkJs --noEmit --allowJs"
}).

%% ============================================
%% INITIALIZATION + BINDING HOOKS
%% ============================================

%% init_annotated_js_target
%  Initialize by delegating to the TypeScript base (bindings, import state).
init_annotated_js_target :-
    init_typescript_target,
    format('[AnnotatedJS Target] JSDoc annotations; tsc --checkJs --noEmit --allowJs~n', []).

clear_binding_imports :-
    typescript_target:clear_binding_imports.

collect_binding_import(Import) :-
    typescript_target:collect_binding_import(Import).

get_collected_imports(Imports) :-
    typescript_target:get_collected_imports(Imports).

%% ============================================
%% COMPILE API (delegate to TS, then rewrite)
%% ============================================

compile_predicate(Pred/Arity, Options, Code) :-
    typescript_target:compile_predicate(Pred/Arity, Options, TSCode),
    ts_to_annotated_js(TSCode, Code),
    !.

compile_facts(Pred, Arity, Code) :-
    typescript_target:compile_facts(Pred, Arity, TSCode),
    ts_to_annotated_js(TSCode, Code),
    !.

compile_recursion(Pred/_Arity, Options, Code) :-
    option(pattern(transitive_closure), Options),
    !,
    annotated_js_tc_from_ts_template(Pred, Options, TSCode),
    ts_to_annotated_js(TSCode, Code).
compile_recursion(Pred/Arity, Options, Code) :-
    typescript_target:compile_recursion(Pred/Arity, Options, TSCode),
    ts_to_annotated_js(TSCode, Code),
    !.

compile_module(Predicates, Options, Code) :-
    typescript_target:compile_module(Predicates, Options, TSCode),
    ts_to_annotated_js(TSCode, Code),
    !.

write_annotated_js_module(Code, Filename) :-
    open(Filename, write, Stream),
    write(Stream, Code),
    close(Stream),
    format('Annotated JS module written to: ~w~n', [Filename]),
    format('Type-check with: npx tsc --checkJs --noEmit --allowJs ~w~n', [Filename]).

%% annotated_js_tc_from_ts_template(+Pred, +Options, -TSCode)
%  Reuse the TypeScript transitive-closure definitions template (do not
%  reimplement BFS). The result is still TypeScript; ts_to_annotated_js/2
%  moves the annotations into JSDoc.
annotated_js_tc_from_ts_template(Pred, Options, TSCode) :-
    atom_string(Pred, PredStr),
    option(base_pred(BasePred), Options, parent),
    atom_string(BasePred, BaseStr),
    (   absolute_file_name(
            'templates/targets/typescript/tc_definitions.mustache',
            Abs, [access(read), file_errors(fail), relative_to('.')])
    ->  read_file_to_string(Abs, Tpl, [])
    ;   module_property(annotated_js_target, file(ThisFile)),
        file_directory_name(ThisFile, TargetDir),
        atom_concat(TargetDir, '/../../../templates/targets/typescript/tc_definitions.mustache', Rel),
        read_file_to_string(Rel, Tpl, [])
    ),
    atomic_list_concat(P1, '{{pred}}', Tpl),
    atomic_list_concat(P1, PredStr, Tpl2),
    atomic_list_concat(P2, '{{base}}', Tpl2),
    atomic_list_concat(P2, BaseStr, TSCode).

%% ============================================
%% TS → ANNOTATED JS (single rewrite predicate)
%% ============================================

%% ts_to_annotated_js(+TSCode, -JSCode)
%  Move inline TypeScript types into JSDoc (`@param`, `@returns`,
%  `@typedef`, `@type`, `@template`), convert `interface` blocks to
%  `@typedef`, and strip type-only syntax so the result is valid
%  ES-module JavaScript. Mirrors clojurescript_interop_rewrite/2: one
%  predicate owns the whole transformation.
ts_to_annotated_js(TSCode, JSCode) :-
    ts_text(TSCode, In),
    split_string(In, "\n", "\r", Lines),
    rewrite_lines(Lines, Rewritten),
    atomic_list_concat(Rewritten, "\n", Joined),
    strip_residual_ts(Joined, Stripped),
    rewrite_banner(Stripped, Bannered),
    js_file_header(Header),
    string_concat(Header, Bannered, JSCode),
    !.

ts_text(Text, String) :-
    (   string(Text)
    ->  String = Text
    ;   atom_string(Text, String)
    ).

js_file_header("/**\n * @file Generated by UnifyWeaver AnnotatedJS Target.\n * Plain JavaScript with JSDoc. Type-check: npx tsc --checkJs --noEmit --allowJs\n */\n").

rewrite_banner(In, Out) :-
    atomic_list_concat(A, 'UnifyWeaver TypeScript Target', In),
    atomic_list_concat(A, 'UnifyWeaver AnnotatedJS Target', S1),
    atomic_list_concat(B, 'TypeScript Target', S1),
    atomic_list_concat(B, 'AnnotatedJS Target', S2),
    atomic_list_concat(C, 'ts-node script.ts', S2),
    atomic_list_concat(C, 'node script.js', S3),
    atomic_list_concat(D, '.ts\n', S3),
    atomic_list_concat(D, '.js\n', Out).

%% ---------- line walker ----------

rewrite_lines([], []).
rewrite_lines([Line|Rest], Out) :-
    interface_start(Line, Name, Templates),
    !,
    take_until_lone_brace(Rest, FieldLines, After),
    interface_to_typedef(Name, Templates, FieldLines, TypedefLines),
    rewrite_lines(After, RestOut),
    append(TypedefLines, RestOut, Out).
rewrite_lines([Line|Rest], Out) :-
    signature_start(Line),
    \+ complete_signature_line(Line),
    !,
    take_until_sig_complete([Line|Rest], SigLines, After),
    atomic_list_concat(SigLines, "\n", SigText),
    rewrite_signature_block(SigText, NewLines),
    rewrite_lines(After, RestOut),
    append(NewLines, RestOut, Out).
rewrite_lines([Line|Rest], OutLines) :-
    rewrite_one_line(Line, NewLines),
    rewrite_lines(Rest, RestOut),
    append(NewLines, RestOut, OutLines).

take_until_lone_brace([], [], []).
take_until_lone_brace([Line|Rest], [], Rest) :-
    normalize_space(string(T), Line),
    ( T == "}" ; T == "};" ),
    !.
take_until_lone_brace([Line|Rest], [Line|Fields], After) :-
    take_until_lone_brace(Rest, Fields, After).

take_until_sig_complete([], [], []).
take_until_sig_complete([Line|Rest], [Line], Rest) :-
    complete_signature_line(Line),
    !.
take_until_sig_complete([Line|Rest], [Line|More], After) :-
    take_until_sig_complete(Rest, More, After).

complete_signature_line(Line) :-
    (   sub_string(Line, _, _, _, "=> {")
    ;   sub_string(Line, _, _, _, "=> {")
    ),
    !.
complete_signature_line(Line) :-
    sub_string(Line, _, _, _, "function "),
    sub_string(Line, _, 1, 0, "{").

signature_start(Line) :-
    normalize_space(string(T), Line),
    (   sub_string(T, 0, _, _, "export const ")
    ;   sub_string(T, 0, _, _, "const ")
    ;   sub_string(T, 0, _, _, "export async function ")
    ;   sub_string(T, 0, _, _, "export function ")
    ;   sub_string(T, 0, _, _, "async function ")
    ;   sub_string(T, 0, _, _, "function ")
    ),
    (   sub_string(T, _, _, _, " = <")
    ;   % `const x = (...)` is only a SIGNATURE when an arrow follows on the
        % same line. Without the `=>` check (G-A3-7) an ordinary generated
        % assignment whose right-hand side is parenthesised --
        %     const v5 = (v4 - arg2);
        % -- was taken for the opening line of a multi-line arrow signature.
        % rewrite_lines/2 then swallowed the whole rest of the file looking for
        % a closing `=> {` that never came and the entire TS->JSDoc rewrite
        % FAILED, so annotated_js refused predicates vanilla_js compiled fine.
        sub_string(T, _, _, _, " = ("),
        sub_string(T, _, _, _, "=>")
    ;   sub_string(T, 0, _, _, "function ")
    ;   sub_string(T, _, _, _, " function ")
    ).

%% ---------- interface → @typedef ----------

interface_start(Line, Name, Templates) :-
    normalize_space(string(T), Line),
    (   sub_string(T, 0, _, After, "export interface ")
    ->  sub_string(T, _, After, 0, Rest)
    ;   sub_string(T, 0, _, After, "interface ")
    ->  sub_string(T, _, After, 0, Rest)
    ),
    parse_interface_head(Rest, Name, Templates).

parse_interface_head(Rest, Name, Templates) :-
    string_chars(Rest, Chars),
    skip_ws(Chars, C1),
    read_ident(C1, NameChars, C2),
    NameChars \= [],
    string_chars(Name, NameChars),
    skip_ws(C2, C3),
    (   C3 = ['<'|C4]
    ->  read_balanced(C4, '<', '>', TplChars, _),
        string_chars(Templates, TplChars)
    ;   Templates = ""
    ).

interface_to_typedef(Name, Templates, FieldLines, [Block]) :-
    template_docs(Templates, TplDocs),
    maplist(field_to_property, FieldLines, PropDocs0),
    exclude(==(""), PropDocs0, PropDocs),
    append(TplDocs, PropDocs, Docs),
    format(string(Head), ' * @typedef {Object} ~w', [Name]),
    atomic_list_concat([Head|Docs], "\n", Inner),
    format(string(Block), '/**\n~w\n */', [Inner]).

template_docs("", []) :- !.
template_docs(Templates, Docs) :-
    split_string(Templates, ",", " \t", Names),
    findall(Doc, (
        member(N, Names),
        N \= "",
        format(string(Doc), ' * @template ~w', [N])
    ), Docs).

field_to_property(Line, "") :-
    normalize_space(string(T), Line),
    ( T == "" ; sub_string(T, 0, 2, _, "//") ),
    !.
field_to_property(Line, Doc) :-
    normalize_space(string(T0), Line),
    (   sub_string(T0, 0, 2, _, "/*")
    ->  fail
    ;   strip_trailing_semi(T0, T)
    ),
    string_chars(T, Chars),
    skip_ws(Chars, C1),
    read_ident(C1, NameChars, C2),
    NameChars \= [],
    string_chars(Name, NameChars),
    skip_ws(C2, C3),
    (   C3 = ['?'|C4]
    ->  Optional = true,
        skip_ws(C4, C5)
    ;   Optional = false,
        C5 = C3
    ),
    C5 = [':'|C6],
    skip_ws(C6, C7),
    read_type(C7, TypeChars, _),
    string_chars(Type0, TypeChars),
    normalize_space(string(Type), Type0),
    (   Optional == true
    ->  format(string(Doc), ' * @property {~w} [~w]', [Type, Name])
    ;   format(string(Doc), ' * @property {~w} ~w', [Type, Name])
    ).

strip_trailing_semi(In, Out) :-
    (   sub_string(In, 0, _, 1, Out0),
        sub_string(In, _, 1, 0, ";")
    ->  normalize_space(string(Out), Out0)
    ;   Out = In
    ).

%% ---------- signatures ----------

rewrite_signature_block(SigText, [JSDoc, Clean]) :-
    split_sig_open(SigText, Sig, Open),
    rewrite_signature(Sig, JSDoc, Clean0),
    string_concat(Clean0, Open, Clean).

rewrite_one_line(Line, [JSDoc, Clean]) :-
    complete_signature_line(Line),
    signature_start(Line),
    !,
    rewrite_signature_block(Line, [JSDoc, Clean]).
rewrite_one_line(Line, Lines) :-
    rewrite_typed_or_generic_binding(Line, Lines),
    !.
rewrite_one_line(Line, [Line]).

split_sig_open(Text, Sig, " {") :-
    sub_string(Text, Before, 2, 0, " {"),
    !,
    sub_string(Text, 0, Before, 2, Sig).
split_sig_open(Text, Sig, "{") :-
    sub_string(Text, Before, 1, 0, "{"),
    !,
    sub_string(Text, 0, Before, 1, Sig).
split_sig_open(Text, Text, "").

%% rewrite_signature(+Sig, -JSDoc, -CleanSig)
rewrite_signature(Sig, JSDoc, CleanSig) :-
    string_chars(Sig, Chars),
    skip_ws(Chars, C0),
    leading_ws(Sig, Indent),
    parse_sig_prefix(C0, Prefix, AfterPrefix),
    (   AfterPrefix = ['<'|CTpl]
    ->  read_balanced(CTpl, '<', '>', TplChars, AfterTpl0),
        string_chars(Templates, TplChars),
        skip_ws(AfterTpl0, AfterTpl)
    ;   Templates = "",
        AfterTpl = AfterPrefix
    ),
    AfterTpl = ['('|ParamChars],
    read_balanced(ParamChars, '(', ')', ParamInner, AfterParams0),
    skip_ws(AfterParams0, AfterParams),
    (   AfterParams = [':'|Ret0]
    ->  skip_ws(Ret0, Ret1),
        read_return_type(Ret1, RetChars, AfterRet),
        string_chars(Ret0s, RetChars),
        normalize_space(string(RetType), Ret0s)
    ;           RetType = "",
        AfterRet = AfterParams
    ),
    string_chars(Tail, AfterRet),
    parse_params(ParamInner, JsParams, ParamDocs),
    atomic_list_concat(JsParams, ', ', JsParamStr),
    template_docs(Templates, TplDocs),
    (   RetType == ""
    ->  RetDocs = []
    ;   format(string(RetDoc), ' * @returns {~w}', [RetType]),
        RetDocs = [RetDoc]
    ),
    append(TplDocs, ParamDocs, D1),
    append(D1, RetDocs, AllDocs),
    (   AllDocs == []
    ->  JSDoc = "/** @file */"
    ;   atomic_list_concat(AllDocs, "\n", Inner),
        format(string(JSDoc), '/**\n~w\n */', [Inner])
    ),
    (   sub_string(Tail, 0, 2, _, "=>")
    ->  format(string(CleanSig), '~w~w(~w) ~w', [Indent, Prefix, JsParamStr, Tail])
    ;   format(string(CleanSig), '~w~w(~w)~w', [Indent, Prefix, JsParamStr, Tail])
    ).

leading_ws(Line, Indent) :-
    string_chars(Line, Chars),
    take_ws(Chars, Ws, _),
    string_chars(Indent, Ws).

take_ws([C|T], [C|W], Rest) :-
    ws_char(C),
    !,
    take_ws(T, W, Rest).
take_ws(Rest, [], Rest).

%% parse_sig_prefix(+Chars, -PrefixAtom, -After)
%  Consume everything up to (but not including) '<' of a generic or '(' of
%  the parameter list. Prefix includes `export const name = ` or `function name`.
parse_sig_prefix(Chars, Prefix, After) :-
    parse_sig_prefix_acc(Chars, [], Prefix, After).

parse_sig_prefix_acc(['<'|T], Acc, Prefix, ['<'|T]) :-
    Acc \= [],
    string_chars(Prefix, Acc),
    !.
parse_sig_prefix_acc(['('|T], Acc, Prefix, ['('|T]) :-
    Acc \= [],
    string_chars(Prefix, Acc),
    !.
parse_sig_prefix_acc([H|T], Acc, Prefix, After) :-
    append(Acc, [H], Acc1),
    parse_sig_prefix_acc(T, Acc1, Prefix, After).

%% ---------- const bindings: `: T =` and `new Map<T>` ----------

rewrite_typed_or_generic_binding(Line, [Doc, Clean]) :-
    typed_binding(Line, Indent, Keywords, Name, Type, Rhs),
    !,
    format(string(Doc), '~w/** @type {~w} */', [Indent, Type]),
    strip_constructor_generics(Rhs, RhsClean),
    format(string(Clean), '~w~w ~w = ~w', [Indent, Keywords, Name, RhsClean]).
rewrite_typed_or_generic_binding(Line, [Doc, Clean]) :-
    generic_ctor_binding(Line, Indent, Keywords, Name, Type, Ctor, Args),
    !,
    format(string(Doc), '~w/** @type {~w} */', [Indent, Type]),
    format(string(Clean), '~w~w ~w = new ~w(~w);', [Indent, Keywords, Name, Ctor, Args]).

typed_binding(Line, Indent, Keywords, Name, Type, Rhs) :-
    leading_ws(Line, Indent),
    normalize_space(string(T), Line),
    binding_keywords(T, Keywords, AfterKw),
    string_chars(AfterKw, Chars),
    skip_ws(Chars, C1),
    read_ident(C1, NameChars, C2),
    NameChars \= [],
    string_chars(Name, NameChars),
    skip_ws(C2, C3),
    C3 = [':'|C4],
    skip_ws(C4, C5),
    read_type(C5, TypeChars, C6),
    string_chars(Type0, TypeChars),
    normalize_space(string(Type), Type0),
    skip_ws(C6, C7),
    C7 = ['='|C8],
    skip_ws(C8, C9),
    string_chars(Rhs0, C9),
    normalize_space(string(Rhs), Rhs0).

generic_ctor_binding(Line, Indent, Keywords, Name, Type, Ctor, Args) :-
    leading_ws(Line, Indent),
    normalize_space(string(T), Line),
    binding_keywords(T, Keywords, AfterKw),
    string_chars(AfterKw, Chars),
    skip_ws(Chars, C1),
    read_ident(C1, NameChars, C2),
    NameChars \= [],
    string_chars(Name, NameChars),
    skip_ws(C2, C3),
    C3 = ['='|C4],
    skip_ws(C4, C5),
    atom_chars('new', NewChars),
    append(NewChars, AfterNew0, C5),
    skip_ws(AfterNew0, AfterNew),
    read_ident(AfterNew, CtorChars, C6),
    string_chars(Ctor, CtorChars),
    ( Ctor == "Map" ; Ctor == "Set" ),
    skip_ws(C6, C7),
    C7 = ['<'|C8],
    read_balanced(C8, '<', '>', TplChars, C9),
    string_chars(Tpl, TplChars),
    format(string(Type), '~w<~w>', [Ctor, Tpl]),
    skip_ws(C9, C10),
    C10 = ['('|C11],
    read_balanced(C11, '(', ')', ArgChars, _),
    string_chars(Args, ArgChars).

binding_keywords(T, "export const", After) :-
    sub_string(T, 0, _, AfterN, "export const "),
    !,
    sub_string(T, _, AfterN, 0, After).
binding_keywords(T, "export let", After) :-
    sub_string(T, 0, _, AfterN, "export let "),
    !,
    sub_string(T, _, AfterN, 0, After).
binding_keywords(T, "const", After) :-
    sub_string(T, 0, _, AfterN, "const "),
    !,
    sub_string(T, _, AfterN, 0, After).
binding_keywords(T, "let", After) :-
    sub_string(T, 0, _, AfterN, "let "),
    !,
    sub_string(T, _, AfterN, 0, After).
binding_keywords(T, "var", After) :-
    sub_string(T, 0, _, AfterN, "var "),
    sub_string(T, _, AfterN, 0, After).

strip_constructor_generics(Rhs, Clean) :-
    strip_new_generics(Rhs, Clean).

%% ---------- params ----------

parse_params([], [], []) :- !.
parse_params(Chars, JsParams, Docs) :-
    skip_ws(Chars, C1),
    C1 \= [],
    next_param(C1, JsParam, Doc, Rest0),
    skip_ws(Rest0, Rest1),
    (   Rest1 = [','|Rest2]
    ->  skip_ws(Rest2, Rest3)
    ;   Rest3 = Rest1
    ),
    parse_params(Rest3, JsRest, DocRest),
    JsParams = [JsParam|JsRest],
    Docs = [Doc|DocRest].

next_param(Chars, JsParam, Doc, Rest) :-
    skip_ws(Chars, C0),
    (   C0 = ['.','.','.'|C1]
    ->  Spread = "...",
        skip_ws(C1, C2)
    ;   Spread = "",
        C2 = C0
    ),
    read_ident(C2, NameChars, C3),
    NameChars \= [],
    string_chars(Name, NameChars),
    skip_ws(C3, C4),
    (   C4 = [':'|C5]
    ->  skip_ws(C5, C6),
        read_type(C6, TypeChars, C7),
        string_chars(Type0, TypeChars),
        normalize_space(string(Type), Type0),
        skip_ws(C7, C8),
        (   default_start(C8, C9)
        ->  skip_ws(C9, C10),
            read_default(C10, DefChars, Rest),
            string_chars(Def0, DefChars),
            normalize_space(string(Default), Def0),
            format(string(JsParam), '~w~w = ~w', [Spread, Name, Default]),
            format(string(Doc), ' * @param {~w} [~w=~w]', [Type, Name, Default])
        ;   Rest = C8,
            format(string(JsParam), '~w~w', [Spread, Name]),
            format(string(Doc), ' * @param {~w} ~w', [Type, Name])
        )
    ;   default_start(C4, C5)
    ->  skip_ws(C5, C6),
        read_default(C6, DefChars, Rest),
        string_chars(Def0, DefChars),
        normalize_space(string(Default), Def0),
        format(string(JsParam), '~w~w = ~w', [Spread, Name, Default]),
        format(string(Doc), ' * @param {*} [~w=~w]', [Name, Default])
    ;   Rest = C4,
        format(string(JsParam), '~w~w', [Spread, Name]),
        format(string(Doc), ' * @param {*} ~w', [Name])
    ).

default_start(['='|T], T) :-
    T \= ['>'|_],
    !.

read_default(Chars, DefChars, Rest) :-
    read_default(Chars, 0, 0, 0, 0, DefChars, Rest).

read_default([], _, _, _, _, [], []) :- !.
read_default([','|T], 0, 0, 0, 0, [], [','|T]) :- !.
read_default([')'|T], 0, 0, 0, 0, [], [')'|T]) :- !.
read_default(['('|T], P, A, B, C, ['('|TT], Rest) :-
    P1 is P + 1, !, read_default(T, P1, A, B, C, TT, Rest).
read_default([')'|T], P, A, B, C, [')'|TT], Rest) :-
    P > 0, P1 is P - 1, !, read_default(T, P1, A, B, C, TT, Rest).
read_default(['<'|T], P, A, B, C, ['<'|TT], Rest) :-
    A1 is A + 1, !, read_default(T, P, A1, B, C, TT, Rest).
read_default(['>'|T], P, A, B, C, ['>'|TT], Rest) :-
    A > 0, A1 is A - 1, !, read_default(T, P, A1, B, C, TT, Rest).
read_default(['['|T], P, A, B, C, ['['|TT], Rest) :-
    B1 is B + 1, !, read_default(T, P, A, B1, C, TT, Rest).
read_default([']'|T], P, A, B, C, [']'|TT], Rest) :-
    B > 0, B1 is B - 1, !, read_default(T, P, A, B1, C, TT, Rest).
read_default(['{'|T], P, A, B, C, ['{'|TT], Rest) :-
    C1 is C + 1, !, read_default(T, P, A, B, C1, TT, Rest).
read_default(['}'|T], P, A, B, C, ['}'|TT], Rest) :-
    C > 0, C1 is C - 1, !, read_default(T, P, A, B, C1, TT, Rest).
read_default([H|T], P, A, B, C, [H|TT], Rest) :-
    read_default(T, P, A, B, C, TT, Rest).

%% ---------- type scanners ----------

read_type(Chars, TypeChars, Rest) :-
    read_type(Chars, 0, 0, 0, 0, TypeChars, Rest).

read_type([], _, _, _, _, [], []) :- !.
read_type(['=','>'|T], P, A, B, C, ['=','>'|TT], Rest) :-
    !,
    read_type(T, P, A, B, C, TT, Rest).
read_type(['='|T], 0, 0, 0, 0, [], ['='|T]) :- !.
read_type([','|T], 0, 0, 0, 0, [], [','|T]) :- !.
read_type([')'|T], 0, 0, 0, 0, [], [')'|T]) :- !.
read_type([';'|T], 0, 0, 0, 0, [], [';'|T]) :- !.
read_type(['('|T], P, A, B, C, ['('|TT], Rest) :-
    P1 is P + 1, !, read_type(T, P1, A, B, C, TT, Rest).
read_type([')'|T], P, A, B, C, [')'|TT], Rest) :-
    P > 0, P1 is P - 1, !, read_type(T, P1, A, B, C, TT, Rest).
read_type(['<'|T], P, A, B, C, ['<'|TT], Rest) :-
    A1 is A + 1, !, read_type(T, P, A1, B, C, TT, Rest).
read_type(['>'|T], P, A, B, C, ['>'|TT], Rest) :-
    A > 0, A1 is A - 1, !, read_type(T, P, A1, B, C, TT, Rest).
read_type(['['|T], P, A, B, C, ['['|TT], Rest) :-
    B1 is B + 1, !, read_type(T, P, A, B1, C, TT, Rest).
read_type([']'|T], P, A, B, C, [']'|TT], Rest) :-
    B > 0, B1 is B - 1, !, read_type(T, P, A, B1, C, TT, Rest).
read_type(['{'|T], P, A, B, C, ['{'|TT], Rest) :-
    C1 is C + 1, !, read_type(T, P, A, B, C1, TT, Rest).
read_type(['}'|T], P, A, B, C, ['}'|TT], Rest) :-
    C > 0, C1 is C - 1, !, read_type(T, P, A, B, C1, TT, Rest).
read_type([H|T], P, A, B, C, [H|TT], Rest) :-
    read_type(T, P, A, B, C, TT, Rest).

read_return_type(Chars, TypeChars, Rest) :-
    read_return_type(Chars, 0, 0, 0, 0, TypeChars, Rest).

read_return_type([], _, _, _, _, [], []) :- !.
read_return_type(['=','>'|T], 0, 0, 0, 0, [], ['=','>'|T]) :- !.
read_return_type(['{'|T], 0, 0, 0, 0, [], ['{'|T]) :- !.
read_return_type(['=','>'|T], P, A, B, C, ['=','>'|TT], Rest) :-
    !,
    read_return_type(T, P, A, B, C, TT, Rest).
read_return_type(['('|T], P, A, B, C, ['('|TT], Rest) :-
    P1 is P + 1, !, read_return_type(T, P1, A, B, C, TT, Rest).
read_return_type([')'|T], P, A, B, C, [')'|TT], Rest) :-
    P > 0, P1 is P - 1, !, read_return_type(T, P1, A, B, C, TT, Rest).
read_return_type(['<'|T], P, A, B, C, ['<'|TT], Rest) :-
    A1 is A + 1, !, read_return_type(T, P, A1, B, C, TT, Rest).
read_return_type(['>'|T], P, A, B, C, ['>'|TT], Rest) :-
    A > 0, A1 is A - 1, !, read_return_type(T, P, A1, B, C, TT, Rest).
read_return_type(['['|T], P, A, B, C, ['['|TT], Rest) :-
    B1 is B + 1, !, read_return_type(T, P, A, B1, C, TT, Rest).
read_return_type([']'|T], P, A, B, C, [']'|TT], Rest) :-
    B > 0, B1 is B - 1, !, read_return_type(T, P, A, B1, C, TT, Rest).
read_return_type(['{'|T], P, A, B, C, ['{'|TT], Rest) :-
    C1 is C + 1, !, read_return_type(T, P, A, B, C1, TT, Rest).
read_return_type(['}'|T], P, A, B, C, ['}'|TT], Rest) :-
    C > 0, C1 is C - 1, !, read_return_type(T, P, A, B, C1, TT, Rest).
read_return_type([H|T], P, A, B, C, [H|TT], Rest) :-
    read_return_type(T, P, A, B, C, TT, Rest).

%% read_balanced(+CharsAfterOpen, +Open, +Close, -InnerChars, -AfterClose)
read_balanced(Chars, Open, Close, Inner, After) :-
    read_balanced(Chars, Open, Close, 1, [], Inner, After).

read_balanced([], _Open0, _Close0, _, Acc, Inner, []) :-
    reverse(Acc, Inner).
read_balanced([Close|T], _Open, Close, 1, Acc, Inner, T) :-
    !,
    reverse(Acc, Inner).
read_balanced([Close|T], Open, Close, D, Acc, Inner, After) :-
    D > 1,
    !,
    D1 is D - 1,
    read_balanced(T, Open, Close, D1, [Close|Acc], Inner, After).
read_balanced([Open|T], Open, Close, D, Acc, Inner, After) :-
    !,
    D1 is D + 1,
    read_balanced(T, Open, Close, D1, [Open|Acc], Inner, After).
read_balanced([H|T], Open, Close, D, Acc, Inner, After) :-
    read_balanced(T, Open, Close, D, [H|Acc], Inner, After).

read_ident([C|T], [C|More], Rest) :-
    ident_start(C),
    !,
    read_ident_rest(T, More, Rest).
read_ident(Rest, [], Rest).

read_ident_rest([C|T], [C|More], Rest) :-
    ident_char(C),
    !,
    read_ident_rest(T, More, Rest).
read_ident_rest(Rest, [], Rest).

ident_start(C) :-
    (   char_type(C, alpha)
    ;   C == '_'
    ;   C == '$'
    ).

ident_char(C) :-
    (   ident_start(C)
    ;   char_type(C, digit)
    ).

skip_ws([C|T], Rest) :-
    ws_char(C),
    !,
    skip_ws(T, Rest).
skip_ws(Rest, Rest).

ws_char(' ').
ws_char('\t').
ws_char('\n').
ws_char('\r').

%% ---------- residual TS-only syntax ----------

strip_residual_ts(In, Out) :-
    strip_as_casts(In, S1),
    strip_nonnull_assertions(S1, S2),
    strip_new_generics(S2, S3),
    strip_inline_arrow_params(S3, Out).

%% (expr as T) → (/** @type {T} */ (expr))
%
%  Only a genuine value-cast `expr as Type` is rewritten. Occurrences of ` as `
%  that are part of an import/export binding — a namespace import
%  `import * as fs from '...'` or an aliased specifier `{ foo as bar }` — are
%  NOT casts and must survive untouched. is_real_as_cast/2 discriminates: a cast
%  has an expression on the left (ends in an identifier char / `)` / `]`, never
%  the `*` of a namespace import) and a type on the right (starts uppercase or
%  with a primitive-type keyword — an import alias like `as fs` / `as bar` does
%  not). Non-cast ` as ` is copied through and scanning continues past it.
strip_as_casts(In, Out) :-
    strip_as_casts_scan(In, "", Out).

strip_as_casts_scan(In, Acc, Out) :-
    (   sub_string(In, Before, 4, _, " as ")
    ->  sub_string(In, 0, Before, _, Left),
        AfterStart is Before + 4,
        sub_string(In, AfterStart, _, 0, Right0),
        (   is_real_as_cast(Left, Right0)
        ->  string_chars(Right0, RChars),
            skip_ws(RChars, RC1),
            read_type(RC1, TypeChars, RC2),
            string_chars(Type0, TypeChars),
            normalize_space(string(Type), Type0),
            string_chars(Left, LChars),
            wrap_as_cast(LChars, Type, CastLeft),
            string_chars(RightRest, RC2),
            string_concat(Acc, CastLeft, Acc1),
            strip_as_casts_scan(RightRest, Acc1, Out)
        ;   string_concat(Acc, Left, Acc1),
            string_concat(Acc1, " as ", Acc2),
            strip_as_casts_scan(Right0, Acc2, Out)
        )
    ;   string_concat(Acc, In, Out)
    ).

%% is_real_as_cast(+Left, +Right)
%  True when the ` as ` between Left and Right is a value cast rather than an
%  import/export alias.
is_real_as_cast(Left, Right) :-
    string_chars(Left, LChars),
    reverse(LChars, LRev),
    skip_ws(LRev, [LastC|_]),
    ( ident_char(LastC) ; LastC == ')' ; LastC == ']' ),
    string_chars(Right, RChars),
    skip_ws(RChars, [FirstC|More]),
    ( char_type(FirstC, upper)
    ; type_keyword_start([FirstC|More])
    ).

%% type_keyword_start(+Chars)
%  Chars begin with a primitive-type keyword as a whole word.
type_keyword_start(Chars) :-
    member(Kw, [any,unknown,number,string,boolean,void,null,undefined,
                object,never,symbol,bigint]),
    atom_chars(Kw, KwChars),
    append(KwChars, Rest, Chars),
    ( Rest == [] ; Rest = [B|_], \+ ident_char(B) ),
    !.

%% strip_inline_arrow_params(+In, -Out)
%  Strip TypeScript type annotations from inline arrow-function parameter lists
%  that the line-level signature parser does not own, e.g. a callback
%  `rl.on("line", (line: string) => {...})` → `(line) => {...}`, and an inline
%  return type `(x): number => x` → `(x) => x`. Only a `(...)` group immediately
%  followed (modulo one return-type annotation) by `=>` is treated as an arrow
%  param list; ordinary calls `foo(a, b)` are left alone. Reuses parse_params/3
%  to reproduce the cleaned, comma-joined parameter names.
strip_inline_arrow_params(In, Out) :-
    string_chars(In, Chars),
    saip(Chars, OutChars),
    string_chars(Out, OutChars).

saip([], []) :- !.
saip(['('|T], Out) :-
    read_balanced(T, '(', ')', Inner, After0),
    arrow_tail(After0, AfterArrow),
    !,
    (   parse_params(Inner, JsParams, _Docs),
        atomic_list_concat(JsParams, ', ', ParamStr)
    ->  string_chars(ParamStr, ParamChars)
    ;   ParamChars = Inner
    ),
    saip(AfterArrow, RestOut),
    append([')',' ','=','>'|RestOut], [], Tail0),
    append(ParamChars, Tail0, Body),
    Out = ['('|Body].
saip([C|T], [C|Out]) :-
    saip(T, Out).

%% arrow_tail(+AfterCloseParen, -AfterArrow)
%  Succeeds when what follows a `)` is (optional `: RetType`) then `=>`;
%  AfterArrow is the remainder after the `=>`.
arrow_tail(After0, AfterArrow) :-
    skip_ws(After0, A1),
    (   A1 = [':'|A2]
    ->  skip_ws(A2, A3),
        read_return_type(A3, _RetChars, A4),
        skip_ws(A4, A5)
    ;   A5 = A1
    ),
    A5 = ['=','>'|AfterArrow].

wrap_as_cast(LChars, Type, CastLeft) :-
    reverse(LChars, Rev),
    skip_ws(Rev, Rev1),
    (   Rev1 = [')'|_]
    ->  % uncommon: already parenthesized oddly
        string_chars(Left, LChars),
        format(string(CastLeft), '~w/** @type {~w} */ ', [Left, Type])
    ;   take_cast_expr(Rev1, ExprRev, BeforeRev),
        reverse(ExprRev, ExprChars),
        reverse(BeforeRev, BeforeChars),
        string_chars(Expr, ExprChars),
        string_chars(Before, BeforeChars),
        format(string(CastLeft), '~w/** @type {~w} */ (~w)', [Before, Type, Expr])
    ).

take_cast_expr([C|T], [C|E], Rest) :-
    ( ident_char(C) ; C == '.' ),
    !,
    take_cast_expr(T, E, Rest).
take_cast_expr(Rest, [], Rest).

%% foo()! / foo]!  → /** @type {*} */ (foo())
strip_nonnull_assertions(In, Out) :-
    (   sub_string(In, Before, 2, After, ")!"),
        \+ bang_continues(After)
    ->  sub_string(In, 0, Before, _, LeftIncl),
        string_concat(LeftIncl, ")", CallToParen),
        wrap_trailing_call(CallToParen, Wrapped),
        sub_string(In, _, After, 0, Right),
        string_concat(Wrapped, Right, Mid),
        strip_nonnull_assertions(Mid, Out)
    ;   sub_string(In, Before, 2, After, "]!"),
        \+ bang_continues(After)
    ->  sub_string(In, 0, Before, _, LeftIncl),
        string_concat(LeftIncl, "]", CallToBrack),
        wrap_trailing_index(CallToBrack, Wrapped),
        sub_string(In, _, After, 0, Right),
        string_concat(Wrapped, Right, Mid),
        strip_nonnull_assertions(Mid, Out)
    ;   Out = In
    ).

bang_continues(After) :-
    sub_string(After, 0, 1, _, Ch),
    member(Ch, ["=", "!"]).

wrap_trailing_call(Text, Wrapped) :-
    string_chars(Text, Chars),
    reverse(Chars, Rev),
    Rev = [')'|R1],
    consume_balanced(R1, ')', '(', 1, Taken, BeforeRev),
    take_callee(BeforeRev, CalleeRev, PrefixRev),
    reverse([')'|Taken], CallChars),
    reverse(CalleeRev, CalleeChars),
    reverse(PrefixRev, PrefixChars),
    append(CalleeChars, CallChars, ExprChars),
    string_chars(Expr, ExprChars),
    string_chars(Prefix, PrefixChars),
    format(string(Wrapped), '~w/** @type {*} */ (~w)', [Prefix, Expr]).

wrap_trailing_index(Text, Wrapped) :-
    string_chars(Text, Chars),
    reverse(Chars, Rev),
    Rev = [']'|R1],
    consume_balanced(R1, ']', '[', 1, Taken, BeforeRev),
    take_callee(BeforeRev, CalleeRev, PrefixRev),
    reverse([']'|Taken], IdxChars),
    reverse(CalleeRev, CalleeChars),
    reverse(PrefixRev, PrefixChars),
    append(CalleeChars, IdxChars, ExprChars),
    string_chars(Expr, ExprChars),
    string_chars(Prefix, PrefixChars),
    format(string(Wrapped), '~w/** @type {*} */ (~w)', [Prefix, Expr]).

consume_balanced([], _Close0, _Open0, _, [], []).
consume_balanced([Close|T], Close, Open, D, [Close|Acc], Rest) :-
    D > 0,
    D1 is D + 1,
    !,
    consume_balanced(T, Close, Open, D1, Acc, Rest).
consume_balanced([Open|T], _Close, Open, 1, [Open], T) :-
    !.
consume_balanced([Open|T], Close, Open, D, [Open|Acc], Rest) :-
    D > 1,
    D1 is D - 1,
    !,
    consume_balanced(T, Close, Open, D1, Acc, Rest).
consume_balanced([H|T], Close, Open, D, [H|Acc], Rest) :-
    consume_balanced(T, Close, Open, D, Acc, Rest).

take_callee([C|T], [C|More], Rest) :-
    ( ident_char(C) ; C == '.' ),
    !,
    take_callee(T, More, Rest).
take_callee(Rest, [], Rest).

%% new Map<K,V>(...) → new Map(...)
strip_new_generics(In, Out) :-
    (   sub_string(In, Before, 8, After, "new Map<")
    ->  sub_string(In, 0, Before, _, Left),
        sub_string(In, _, After, 0, Right0),
        drop_generic_args(Right0, Right),
        string_concat(Left, "new Map", Mid0),
        string_concat(Mid0, Right, Mid),
        strip_new_generics(Mid, Out)
    ;   sub_string(In, Before, 8, After, "new Set<")
    ->  sub_string(In, 0, Before, _, Left),
        sub_string(In, _, After, 0, Right0),
        drop_generic_args(Right0, Right),
        string_concat(Left, "new Set", Mid0),
        string_concat(Mid0, Right, Mid),
        strip_new_generics(Mid, Out)
    ;   Out = In
    ).

drop_generic_args(Right0, Right) :-
    string_chars(Right0, Chars),
    read_balanced(Chars, '<', '>', _, After),
    string_chars(Right, After).

%% ============================================
%% ADVANCED RECURSION — inherit TS, then annotate
%% ============================================

:- multifile tail_recursion:compile_tail_pattern/9.
:- multifile linear_recursion:compile_linear_pattern/8.
:- multifile tree_recursion:compile_tree_pattern/6.
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.
:- multifile mutual_recursion:compile_mutual_pattern/5.
:- multifile advanced_recursive_compiler:compile_general_recursive_pattern/6.

tail_recursion:compile_tail_pattern(annotated_js, PredStr, Arity, Base, Rec, AccPos, StepOp, Exit, Code) :-
    tail_recursion:compile_tail_pattern(typescript, PredStr, Arity, Base, Rec, AccPos, StepOp, Exit, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, Code).

linear_recursion:compile_linear_pattern(annotated_js, PredStr, Arity, Base, Rec, Memo, Strat, Code) :-
    linear_recursion:compile_linear_pattern(typescript, PredStr, Arity, Base, Rec, Memo, Strat, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, Code).

tree_recursion:compile_tree_pattern(annotated_js, Pattern, Pred, Arity, UseMemo, Code) :-
    tree_recursion:compile_tree_pattern(typescript, Pattern, Pred, Arity, UseMemo, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, Code).

multicall_linear_recursion:compile_multicall_pattern(annotated_js, PredStr, Base, Rec, Memo, Code) :-
    multicall_linear_recursion:compile_multicall_pattern(typescript, PredStr, Base, Rec, Memo, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, Code).

direct_multi_call_recursion:compile_direct_multicall_pattern(annotated_js, PredStr, Base, Rec, Code) :-
    direct_multi_call_recursion:compile_direct_multicall_pattern(typescript, PredStr, Base, Rec, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, Code).

mutual_recursion:compile_mutual_pattern(annotated_js, Preds, Memo, Strat, Code) :-
    mutual_recursion:compile_mutual_pattern(typescript, Preds, Memo, Strat, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, Code).

advanced_recursive_compiler:compile_general_recursive_pattern(annotated_js, PredStr, Arity, Base, Rec, Code) :-
    advanced_recursive_compiler:compile_general_recursive_pattern(typescript, PredStr, Arity, Base, Rec, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, Code).
