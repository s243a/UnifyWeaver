% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_parser, [
    plawk_parse_string/2
]).

%% plawk_parse_string(+Source, -Program) is semidet.
%
%  Parse the first Phase-2 surface slice:
%
%      /^PREFIX/ { print $0 }
%      $N == "VALUE" { print $0 }
%
%  The AST is deliberately small and explicit so later syntax can extend it
%  without changing the native codegen contract.
plawk_parse_string(Source, Program) :-
    string(Source),
    string_codes(Source, Codes),
    phrase(plawk_program(Program), Codes).

plawk_program(program([], [rule(Pattern, [print(field(0))])], [])) -->
    ws,
    pattern(Pattern),
    ws,
    "{",
    ws,
    print_field_zero,
    ws,
    "}",
    ws,
    eos.

pattern(Pattern) -->
    prefix_pattern(Pattern),
    !.
pattern(Pattern) -->
    field_eq_pattern(Pattern).

prefix_pattern(prefix(Prefix)) -->
    "/^",
    prefix_codes(Codes),
    "/",
    { Codes \== [],
      string_codes(Prefix, Codes)
    }.

field_eq_pattern(field_eq(Index, Value)) -->
    "$",
    integer_codes(IndexCodes),
    ws,
    "==",
    ws,
    quoted_string(ValueCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index > 0,
      ValueCodes \== [],
      string_codes(Value, ValueCodes)
    }.

prefix_codes([Code | Codes]) -->
    [Code],
    { Code =\= 0'/,
      Code =\= 0'\n,
      Code =\= 0'\r
    },
    prefix_codes_rest(Codes).

prefix_codes_rest([Code | Codes]) -->
    [Code],
    { Code =\= 0'/,
      Code =\= 0'\n,
      Code =\= 0'\r
    },
    !,
    prefix_codes_rest(Codes).
prefix_codes_rest([]) -->
    [].

integer_codes([Code | Codes]) -->
    [Code],
    { code_type(Code, digit) },
    integer_codes_rest(Codes).

integer_codes_rest([Code | Codes]) -->
    [Code],
    { code_type(Code, digit) },
    !,
    integer_codes_rest(Codes).
integer_codes_rest([]) -->
    [].

quoted_string(Codes) -->
    "\"",
    quoted_string_codes(Codes),
    "\"".

quoted_string_codes([Code | Codes]) -->
    [Code],
    { Code =\= 0'", Code =\= 0'\n, Code =\= 0'\r },
    !,
    quoted_string_codes(Codes).
quoted_string_codes([]) -->
    [].

print_field_zero -->
    "print",
    required_ws,
    "$0".

required_ws -->
    [Code],
    { code_type(Code, space) },
    ws.

ws -->
    [Code],
    { code_type(Code, space) },
    !,
    ws.
ws -->
    [].

eos([], []).
