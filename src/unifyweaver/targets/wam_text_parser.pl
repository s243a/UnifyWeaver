:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_text_parser.pl - Shared WAM-text parser building blocks
%
% Per WAM_ITEMS_API_PHILOSOPHY.md and WAM_ITEMS_API_SPECIFICATION.md
% (PR #2086): each WAM target used to ship its own tokenizer +
% recogniser, with subtle bugs that drifted across targets (the
% quote-aware tokenizer fix from PR #2076 lived only in the C++
% target until this module landed; the ,/2 fix from PR #2084 same).
% This module is the single source of truth.
%
% Public predicates:
%   wam_tokenize_line/2        - quote-aware tokenizer (whitespace-
%                                separated, with bare-comma logic that
%                                preserves ",/N"-shaped functor names).
%   wam_recognise_label/2      - one-token line ending in ':' → label
%                                name string.
%   wam_recognise_instruction/2 - token list → standard WAM
%                                instruction term, or fail.
%   wam_text_to_items/2        - high-level: WAM text → list of
%                                label(Name) and instruction items in
%                                source order. Unrecognised lines
%                                silently skipped (the long-standing
%                                tolerance behaviour of every target''s
%                                parser).
%
% The instruction-term shapes are the catalogue from
% WAM_ITEMS_API_SPECIFICATION.md §2 — every target consumes them
% the same way.

:- module(wam_text_parser, [
    wam_tokenize_line/2,           % +Line, -Tokens
    wam_recognise_label/2,         % +Tokens, -LabelName
    wam_recognise_instruction/2,   % +Tokens, -Item
    wam_text_to_items/2            % +WamText, -Items
]).

:- use_module(library(lists)).

% =====================================================================
% wam_text_to_items/2
% =====================================================================

%% wam_text_to_items(+WamText, -Items) is det.
%
%  Items is a list interleaving label(NameStr) and instruction terms
%  in source order. Anything that fails to recognise as either is
%  silently skipped — keeps the format forward-compatible with new
%  instructions that haven''t been added to wam_recognise_instruction
%  yet (and matches every target''s historical tolerance behaviour).
wam_text_to_items(WamText, Items) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines(Lines, Items).

parse_lines([], []).
parse_lines([Line|Rest], Items) :-
    wam_tokenize_line(Line, Tokens),
    (   Tokens == []
    ->  parse_lines(Rest, Items)
    ;   wam_recognise_label(Tokens, Name)
    ->  Items = [label(Name)|More],
        parse_lines(Rest, More)
    ;   wam_recognise_instruction(Tokens, Instr)
    ->  Items = [Instr|More],
        parse_lines(Rest, More)
    ;   parse_lines(Rest, Items)
    ).

% =====================================================================
% Tokenizer
% =====================================================================

%% wam_tokenize_line(+Line, -Tokens) is det.
%
%  Splits a WAM-text line into tokens. Whitespace separates;
%  single-quoted regions preserve their content verbatim (with the
%  surrounding quotes stripped); bare commas are separators except
%  when immediately followed by ''/'' (so the conjunction functor
%  ",/N" survives as one token). Inside quotes, backslash escapes the
%  following character, matching wam_target:quote_wam_constant/2.
wam_tokenize_line(Line, Tokens) :-
    string_chars(Line, Chars),
    tokenize_chars(Chars, Tokens).

tokenize_chars([], []).
tokenize_chars([C|Cs], Tokens) :-
    (   ws(C)
    ->  tokenize_chars(Cs, Tokens)
    ;   C == '\''
    ->  read_quoted(Cs, QChars, Rest),
        string_chars(Tok, QChars),
        Tokens = [Tok|More],
        tokenize_chars(Rest, More)
    ;   C == ',' , \+ ( Cs = ['/'|_] )
    ->  % Bare comma (not part of ",/N") is the WAM printer''s
        % argument separator — discard.
        tokenize_chars(Cs, Tokens)
    ;   read_unquoted([C|Cs], TChars, Rest),
        string_chars(Tok, TChars),
        Tokens = [Tok|More],
        tokenize_chars(Rest, More)
    ).

ws(' ').
ws('\t').

read_quoted([], [], []).
read_quoted(['\''|Rest], [], Rest) :- !.
read_quoted(['\\', C|Cs], [C|More], Rest) :- !,
    read_quoted(Cs, More, Rest).
read_quoted([C|Cs], [C|More], Rest) :-
    read_quoted(Cs, More, Rest).

read_unquoted([], [], []).
read_unquoted([C|Cs], [], [C|Cs]) :- ws(C), !.
read_unquoted([',' | Cs], [], [',' | Cs]) :-
    \+ ( Cs = ['/'|_] ), !.
read_unquoted([C|Cs], [C|More], Rest) :-
    read_unquoted(Cs, More, Rest).

% =====================================================================
% Label / instruction recognisers
% =====================================================================

%% wam_recognise_label(+Tokens, -LabelName) is semidet.
%
%  Recognises a label line. A label is a single token ending in '':''.
%  Strips the colon, returns the LabelName as a string.
wam_recognise_label([First|_], LabelName) :-
    sub_string(First, _, 1, 0, ":"),
    string_length(First, Len),
    L1 is Len - 1,
    sub_string(First, 0, L1, _, LabelName).

%% wam_recognise_instruction(+Tokens, -Item) is semidet.
%
%  Recognises a standard WAM instruction line. Returns the
%  corresponding item term per the catalogue in
%  WAM_ITEMS_API_SPECIFICATION.md §2. Fails (no exception) on
%  unrecognised tokens — caller can chain its own
%  extension-recogniser.
wam_recognise_instruction(["get_constant", C, Ai],     get_constant(C, Ai)).
wam_recognise_instruction(["get_variable", Xn, Ai],    get_variable(Xn, Ai)).
wam_recognise_instruction(["get_value", Xn, Ai],       get_value(Xn, Ai)).
wam_recognise_instruction(["get_structure", F, Ai],    get_structure(F, Ai)).
wam_recognise_instruction(["get_list", Ai],            get_list(Ai)).
wam_recognise_instruction(["get_nil", Ai],             get_nil(Ai)).
wam_recognise_instruction(["get_integer", N, Ai],      get_integer(N, Ai)).
wam_recognise_instruction(["unify_variable", Xn],      unify_variable(Xn)).
wam_recognise_instruction(["unify_value", Xn],         unify_value(Xn)).
wam_recognise_instruction(["unify_constant", C],       unify_constant(C)).
wam_recognise_instruction(["put_variable", Xn, Ai],    put_variable(Xn, Ai)).
wam_recognise_instruction(["put_value", Xn, Ai],       put_value(Xn, Ai)).
wam_recognise_instruction(["put_constant", C, Ai],     put_constant(C, Ai)).
wam_recognise_instruction(["put_structure", F, Ai],    put_structure(F, Ai)).
wam_recognise_instruction(["put_list", Ai],            put_list(Ai)).
wam_recognise_instruction(["set_variable", Xn],        set_variable(Xn)).
wam_recognise_instruction(["set_value", Xn],           set_value(Xn)).
wam_recognise_instruction(["set_constant", C],         set_constant(C)).
wam_recognise_instruction(["call", P, N],              call(P, N)).
wam_recognise_instruction(["execute", P],              execute(P)).
wam_recognise_instruction(["proceed"],                 proceed).
wam_recognise_instruction(["fail"],                    fail).
wam_recognise_instruction(["allocate"],                allocate).
wam_recognise_instruction(["deallocate"],              deallocate).
wam_recognise_instruction(["builtin_call", Op, Ar],    builtin_call(Op, Ar)).
wam_recognise_instruction(["call_foreign", Pred, Ar],  call_foreign(Pred, Ar)).
wam_recognise_instruction(["try_me_else", L],          try_me_else(L)).
wam_recognise_instruction(["retry_me_else", L],        retry_me_else(L)).
wam_recognise_instruction(["trust_me"],                trust_me).
wam_recognise_instruction(["jump", L],                 jump(L)).
wam_recognise_instruction(["cut_ite"],                 cut_ite).
wam_recognise_instruction(["begin_aggregate", K, V, R], begin_aggregate(K, V, R)).
wam_recognise_instruction(["begin_aggregate", K, V, R, W],
                                                       begin_aggregate(K, V, R, W)).
wam_recognise_instruction(["end_aggregate", R],        end_aggregate(R)).
% Indexing instructions: variable-arity tail tokens captured as a list,
% target-side decoders parse them per the dispatch-table convention.
wam_recognise_instruction(["switch_on_constant"     | Es], switch_on_constant(Es)).
wam_recognise_instruction(["switch_on_constant_a2"  | Es], switch_on_constant_a2(Es)).
wam_recognise_instruction(["switch_on_structure"    | Es], switch_on_structure(Es)).
wam_recognise_instruction(["switch_on_term"         | Ts], switch_on_term(Ts)).
