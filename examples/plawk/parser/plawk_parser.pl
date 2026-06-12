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
%      $N == "VALUE" { print $M, $K }
%
%  The AST is deliberately small and explicit so later syntax can extend it
%  without changing the native codegen contract.
plawk_parse_string(Source, Program) :-
    string(Source),
    string_codes(Source, Codes),
    phrase(plawk_program(Program), Codes).

plawk_program(program([], [rule(Pattern, [PrintAction])], [])) -->
    ws,
    pattern(Pattern),
    ws,
    "{",
    ws,
    print_action(PrintAction),
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

print_action(print(Fields)) -->
    "print",
    required_ws,
    print_fields(Fields).

print_fields([Field | Fields]) -->
    field_expr(Field),
    print_fields_rest(Fields).

print_fields_rest([Field | Fields]) -->
    ws,
    ",",
    ws,
    !,
    field_expr(Field),
    print_fields_rest(Fields).
print_fields_rest([]) -->
    [].

field_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.

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
