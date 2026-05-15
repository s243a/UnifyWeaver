:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Unit tests for src/unifyweaver/targets/wam_text_parser.pl.
%
% Pins the tokenizer + recogniser shapes that every WAM target now
% relies on (per WAM_ITEMS_API_SPECIFICATION §2). Two regression
% guards in particular: PR #2076''s quoted-atom handling and
% PR #2084''s ",/N"-shaped operator handling — both used to live
% only in the C++ target''s private parser and silently failed in
% the lowered emitter''s copy.

:- use_module('../src/unifyweaver/targets/wam_text_parser').
:- use_module(library(plunit)).

:- begin_tests(wam_text_parser).

% ------------------------------------------------------------------
% wam_tokenize_line/2
% ------------------------------------------------------------------

test(tokenize_simple_instruction) :-
    wam_tokenize_line("    put_variable X1, A1", T),
    assertion(T == ["put_variable", "X1", "A1"]).

test(tokenize_label_keeps_colon) :-
    wam_tokenize_line("L_my_member_2_2:", T),
    assertion(T == ["L_my_member_2_2:"]).

test(tokenize_quoted_atom_with_spaces) :-
    % PR #2076 regression: format strings must survive tokenization
    % as a single token.
    wam_tokenize_line("    put_constant 'plain text~n', A1", T),
    assertion(T == ["put_constant", "plain text~n", "A1"]).

test(tokenize_quoted_atom_with_internal_commas) :-
    wam_tokenize_line("    put_constant 'a, b, c', A1", T),
    assertion(T == ["put_constant", "a, b, c", "A1"]).

test(tokenize_quoted_atom_with_escaped_quote) :-
    wam_tokenize_line("    put_constant 'it\\'s', A1", T),
    assertion(T == ["put_constant", "it's", "A1"]).

test(tokenize_quoted_atom_with_escaped_backslash) :-
    wam_tokenize_line("    put_constant 'a\\\\b', A1", T),
    assertion(T == ["put_constant", "a\\b", "A1"]).

test(tokenize_conjunction_functor_preserved) :-
    % PR #2084 regression: ",/2" must be one token; bare comma
    % between arg positions should still split.
    wam_tokenize_line("    put_structure ,/2, A1", T),
    assertion(T == ["put_structure", ",/2", "A1"]).

test(tokenize_negative_integer_constant) :-
    wam_tokenize_line("    set_constant -1", T),
    assertion(T == ["set_constant", "-1"]).

test(tokenize_no_args) :-
    wam_tokenize_line("    proceed", T),
    assertion(T == ["proceed"]).

test(tokenize_blank_line) :-
    wam_tokenize_line("", T),
    assertion(T == []).

test(tokenize_whitespace_only) :-
    wam_tokenize_line("      \t  ", T),
    assertion(T == []).

% ------------------------------------------------------------------
% wam_recognise_label/2
% ------------------------------------------------------------------

test(recognise_label_basic) :-
    wam_recognise_label(["my_pred/2:"], N),
    assertion(N == "my_pred/2").

test(recognise_label_with_underscores) :-
    wam_recognise_label(["L_clause_3:"], N),
    assertion(N == "L_clause_3").

test(recognise_label_rejects_non_label, [fail]) :-
    wam_recognise_label(["proceed"], _).

test(recognise_label_rejects_instruction_args, [fail]) :-
    wam_recognise_label(["put_variable", "X1", "A1"], _).

% ------------------------------------------------------------------
% wam_recognise_instruction/2
% ------------------------------------------------------------------

test(recognise_get_constant) :-
    wam_recognise_instruction(["get_constant", "5", "A1"], I),
    assertion(I == get_constant("5", "A1")).

test(recognise_put_variable) :-
    wam_recognise_instruction(["put_variable", "Y1", "A1"], I),
    assertion(I == put_variable("Y1", "A1")).

test(recognise_proceed) :-
    wam_recognise_instruction(["proceed"], I),
    assertion(I == proceed).

test(recognise_call) :-
    wam_recognise_instruction(["call", "foo/2", "2"], I),
    assertion(I == call("foo/2", "2")).

test(recognise_execute) :-
    wam_recognise_instruction(["execute", "catch/3"], I),
    assertion(I == execute("catch/3")).

test(recognise_builtin_call) :-
    wam_recognise_instruction(["builtin_call", "is/2", "2"], I),
    assertion(I == builtin_call("is/2", "2")).

test(recognise_try_me_else) :-
    wam_recognise_instruction(["try_me_else", "L_clause_2"], I),
    assertion(I == try_me_else("L_clause_2")).

test(recognise_switch_on_constant_captures_tail) :-
    % Variable-arity instruction — entries captured as one list,
    % parsed by the per-target decoder.
    wam_recognise_instruction(
        ["switch_on_constant", "a:L1", "b:L2"], I),
    assertion(I == switch_on_constant(["a:L1", "b:L2"])).

test(recognise_unknown_instruction_fails, [fail]) :-
    wam_recognise_instruction(["unknown_op", "X"], _).

% ------------------------------------------------------------------
% wam_text_to_items/2 — end-to-end
% ------------------------------------------------------------------

test(text_to_items_single_clause) :-
    Text = "foo/0:\n    proceed\n",
    wam_text_to_items(Text, Items),
    assertion(Items == [label("foo/0"), proceed]).

test(text_to_items_with_quoted_atom) :-
    % End-to-end check that the format-string fix works through the
    % full pipeline (regression guard for PR #2076).
    Text = "tp/0:\n    put_constant 'hello world', A1\n    execute write/1\n",
    wam_text_to_items(Text, Items),
    assertion(Items == [label("tp/0"),
                        put_constant("hello world", "A1"),
                        execute("write/1")]).

test(text_to_items_with_conjunction_functor) :-
    % End-to-end regression guard for PR #2084.
    Text = "tp/0:\n    put_structure ,/2, A1\n    proceed\n",
    wam_text_to_items(Text, Items),
    assertion(Items == [label("tp/0"),
                        put_structure(",/2", "A1"),
                        proceed]).

test(text_to_items_skips_unrecognised) :-
    % Forward-compat: unknown instruction lines silently dropped, the
    % rest of the predicate parses fine.
    Text = "tp/0:\n    proceed\n    unknown_future_op X1\n    fail\n",
    wam_text_to_items(Text, Items),
    assertion(Items == [label("tp/0"), proceed, fail]).

:- end_tests(wam_text_parser).
