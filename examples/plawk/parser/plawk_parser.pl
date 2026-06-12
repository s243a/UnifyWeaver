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
%
%  The AST is deliberately small and explicit so later syntax can extend it
%  without changing the native codegen contract.
plawk_parse_string(Source, Program) :-
    string(Source),
    string_codes(Source, Codes),
    phrase(plawk_program(Program), Codes).

plawk_program(program([], [rule(prefix(Prefix), [print(field(0))])], [])) -->
    ws,
    prefix_pattern(Prefix),
    ws,
    "{",
    ws,
    print_field_zero,
    ws,
    "}",
    ws,
    eos.

prefix_pattern(Prefix) -->
    "/^",
    prefix_codes(Codes),
    "/",
    { Codes \== [],
      string_codes(Prefix, Codes)
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
