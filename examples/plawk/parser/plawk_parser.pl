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
%      /LITERAL/ { print $0 }
%      $N == "VALUE" { print $0 }
%      $N == "VALUE" { print $M, $K }
%      $N == "VALUE" { count++ } END { print count }
%      $N == "VALUE" { errors++; matches++ } END { print errors, matches }
%      $N == "ERROR" { errors++ } $N == "WARN" { warnings++ } END { print errors, warnings }
%      { counts[$1]++ } END { print counts["ERROR"], counts["WARN"] }
%      BEGIN { print "kind", "count" } { count++ } END { print "count", count }
%      BEGIN { FS = ":" } $1 == "ERROR" { counts[$2]++ } END { print counts["disk"] }
%      BEGIN { FS = ":"; OFS = "," } $1 == "ERROR" { print $2, $3 }
%      $1 == "ERROR" { printf "%s=%s\n", $2, $3 }
%      { count++ } END { print "count", count }
%      $3 > 100 { big++ } END { print big }
%      $1 == "ERROR" { print $3, int($3) }
%      $1 == "ERROR" { print int($3) + 1 }
%      $1 == "ERROR" { print int($3) - 1 }
%      $1 == "ERROR" { print NR - 1, NF + 1, length($0) - 3, index($2, "sk") + 1 }
%      $1 == "ERROR" { bytes += $3; last = $3 } END { print bytes, last }
%      $1 == "ERROR" { bytes += length($0); hits += 2 } END { print bytes, hits }
%      { last_pos = index($2, "sk") + 1; total_pos += index($0, "disk") - 1 } END { print last_pos, total_pos }
%      { adjusted += length($0) - 3; width = NF; fields += NF } END { print adjusted, width, fields }
%      { last = NR; prev = NR - 1; total += NR + 1 } END { print last, prev, total }
%      $1 == "DEBUG" { skipped++; next } { total++ } END { print total, skipped }
%      $1 == "ERROR" { hits++; break } { total++ } END { print hits, total }
%      $1 == "ERROR" { last_len = length($0); hits++ } END { print hits, last_len }
%      { if ($1 == "ERROR") { errors++ } else { warnings++ } } END { print errors, warnings }
%
%  The AST is deliberately small and explicit so later syntax can extend it
%  without changing the native codegen contract.
plawk_parse_string(Source, Program) :-
    string(Source),
    string_codes(Source, Codes),
    phrase(plawk_program(Program), Codes).

plawk_program(program(BeginClauses, Rules, EndClauses)) -->
    ws,
    begin_clauses(BeginClauses),
    rules(Rules),
    end_clauses(EndClauses),
    eos.

rules([Rule | Rules]) -->
    rule(Rule),
    rules_rest(Rules).

rules_rest([Rule | Rules]) -->
    ws,
    rule(Rule),
    !,
    rules_rest(Rules).
rules_rest([]) -->
    ws.

rule(rule(Pattern, Actions)) -->
    pattern(Pattern),
    !,
    ws,
    action_block(Actions).
rule(rule(always, Actions)) -->
    action_block(Actions).

action_block(Actions) -->
    "{",
    ws,
    actions(Actions),
    ws,
    "}",
    ws.

%% pattern(-Pattern)//
%
%  awk pattern combinators: `!` binds tighter than `&&`, which binds
%  tighter than `||`; both binary forms associate left and parentheses
%  group. The base patterns are the existing prefix, contains,
%  numeric-compare, and field-equality guards.
pattern(Pattern) -->
    or_pattern(Pattern).

or_pattern(Pattern) -->
    and_pattern(First),
    or_pattern_chain(First, Pattern).

or_pattern_chain(Acc, Pattern) -->
    ws,
    "||",
    ws,
    and_pattern(Right),
    !,
    or_pattern_chain(or_pat(Acc, Right), Pattern).
or_pattern_chain(Pattern, Pattern) -->
    [].

and_pattern(Pattern) -->
    not_pattern(First),
    and_pattern_chain(First, Pattern).

and_pattern_chain(Acc, Pattern) -->
    ws,
    "&&",
    ws,
    not_pattern(Right),
    !,
    and_pattern_chain(and_pat(Acc, Right), Pattern).
and_pattern_chain(Pattern, Pattern) -->
    [].

not_pattern(not_pat(Pattern)) -->
    "!",
    ws,
    not_pattern(Pattern),
    !.
not_pattern(Pattern) -->
    "(",
    ws,
    or_pattern(Pattern),
    ws,
    ")",
    !.
not_pattern(Pattern) -->
    base_pattern(Pattern).

base_pattern(Pattern) -->
    slash_regex_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    field_match_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    field_i64_cmp_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    field_eq_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    prolog_guard_pattern(Pattern).

%% prolog_guard_pattern(-Pattern)//
%
%  A named Prolog predicate as a rule guard: `pred(args...)` matches
%  when the compiled predicate succeeds. Arguments are field atoms
%  ($0 is the whole record), string literal atoms, or integers.
prolog_guard_pattern(prolog_guard(Name, Args)) -->
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")".

foreign_args([Arg | Args]) -->
    foreign_arg(Arg),
    foreign_args_rest(Args).

foreign_args_rest([Arg | Args]) -->
    ws,
    ",",
    ws,
    !,
    foreign_arg(Arg),
    foreign_args_rest(Args).
foreign_args_rest([]) -->
    [].

foreign_arg(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    !,
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
foreign_arg(string(Value)) -->
    quoted_string(ValueCodes),
    !,
    { string_codes(Value, ValueCodes) }.
foreign_arg(int(Value)) -->
    signed_integer_value(Value).

%% slash_regex_pattern(-Pattern)//
%
%  A bare /re/ pattern matches the whole record, as in awk. Bodies with
%  no ERE metacharacters keep their existing fast native lowerings: a
%  leading ^ plus a literal rest stays prefix/1 and an all-literal body
%  stays contains/1. Anything else becomes field_match(0, Regex) and is
%  matched with POSIX ERE at runtime.
slash_regex_pattern(Pattern) -->
    "/",
    regex_body_codes(Codes),
    "/",
    { Codes \== [],
      classify_regex_codes(Codes, Pattern)
    }.

classify_regex_codes([0'^ | Rest], prefix(Prefix)) :-
    Rest \== [],
    \+ ere_metachar_in_codes(Rest),
    !,
    string_codes(Prefix, Rest).
classify_regex_codes(Codes, contains(Literal)) :-
    \+ ere_metachar_in_codes(Codes),
    !,
    string_codes(Literal, Codes).
classify_regex_codes(Codes, field_match(0, Regex)) :-
    string_codes(Regex, Codes).

ere_metachar_in_codes(Codes) :-
    member(Code, Codes),
    memberchk(Code, [0'., 0'[, 0'], 0'(, 0'), 0'*, 0'+, 0'?,
                     0'{, 0'}, 0'|, 0'^, 0'$, 0'\\]),
    !.

%% field_match_pattern(-Pattern)//
%
%  awk's match operators: $N ~ /re/ and $N !~ /re/. $0 works via
%  field index 0. !~ reuses the combinator AST as not_pat(field_match).
field_match_pattern(Pattern) -->
    "$",
    integer_codes(IndexCodes),
    ws,
    match_operator(Negated),
    ws,
    "/",
    regex_body_codes(Codes),
    "/",
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0,
      Codes \== [],
      string_codes(Regex, Codes),
      (   Negated == false
      ->  Pattern = field_match(Index, Regex)
      ;   Pattern = not_pat(field_match(Index, Regex))
      )
    }.

match_operator(true) -->
    "!~".
match_operator(false) -->
    "~".

%% regex_body_codes(-Codes)//
%
%  Codes between the pattern slashes. Backslash pairs pass through
%  unchanged so ERE escapes like \. reach regcomp, except \/ which
%  unescapes to a literal slash.
regex_body_codes([Code | Codes]) -->
    regex_body_code(Code),
    regex_body_codes_rest(Codes).

regex_body_codes_rest(Codes) -->
    regex_body_code(Code),
    !,
    { Codes = [Code | Rest] },
    regex_body_codes_rest(Rest).
regex_body_codes_rest([]) -->
    [].

regex_body_code(0'/) -->
    "\\/",
    !.
regex_body_code(Code) -->
    [Code],
    { Code =\= 0'/,
      Code =\= 0'\n,
      Code =\= 0'\r
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

field_i64_cmp_pattern(field_cmp(Index, Op, Value)) -->
    "$",
    integer_codes(IndexCodes),
    ws,
    numeric_cmp_op(Op),
    ws,
    signed_integer_value(Value),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index > 0
    }.

numeric_cmp_op(eq) -->
    "==".
numeric_cmp_op(ne) -->
    "!=".
numeric_cmp_op(le) -->
    "<=".
numeric_cmp_op(ge) -->
    ">=".
numeric_cmp_op(lt) -->
    "<".
numeric_cmp_op(gt) -->
    ">".


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

signed_integer_value(Value) -->
    "-",
    !,
    integer_codes(Digits),
    { Digits \== [],
      number_codes(Magnitude, Digits),
      Value is -Magnitude
    }.
signed_integer_value(Value) -->
    "+",
    !,
    integer_codes(Digits),
    { Digits \== [],
      number_codes(Value, Digits)
    }.
signed_integer_value(Value) -->
    integer_codes(Digits),
    { Digits \== [],
      number_codes(Value, Digits)
    }.

quoted_string(Codes) -->
    "\"",
    quoted_string_codes(Codes),
    "\"".

quoted_string_codes(Codes) -->
    "\\",
    quoted_string_escape_codes(EscapedCodes),
    !,
    quoted_string_codes(RestCodes),
    { append(EscapedCodes, RestCodes, Codes) }.
quoted_string_codes([Code | Codes]) -->
    [Code],
    { Code =\= 0'", Code =\= 0'\n, Code =\= 0'\r },
    !,
    quoted_string_codes(Codes).
quoted_string_codes([]) -->
    [].

quoted_string_escape_codes([10]) -->
    "n".
quoted_string_escape_codes([9]) -->
    "t".
quoted_string_escape_codes([13]) -->
    "r".
quoted_string_escape_codes([0'"]) -->
    "\"".
quoted_string_escape_codes([0'\\]) -->
    "\\".
quoted_string_escape_codes([0'\\, Code]) -->
    [Code],
    { Code =\= 0'\n, Code =\= 0'\r }.

begin_clauses([begin(Actions)]) -->
    "BEGIN",
    ws,
    "{",
    ws,
    begin_actions(Actions),
    ws,
    "}",
    ws,
    !.
begin_clauses([]) -->
    [].

begin_actions([Action | Actions]) -->
    begin_action(Action),
    begin_actions_rest(Actions).

begin_actions_rest([Action | Actions]) -->
    ws,
    ";",
    ws,
    !,
    begin_action(Action),
    begin_actions_rest(Actions).
begin_actions_rest([]) -->
    [].

begin_action(Action) -->
    begin_assignment(Action),
    !.
begin_action(Action) -->
    print_action(Action),
    !.

begin_assignment(set(var(Name), string(Value))) -->
    begin_assignment_name(Name),
    ws,
    "=",
    ws,
    quoted_string(ValueCodes),
    { string_codes(Value, ValueCodes) }.

begin_assignment_name('BINFMT') -->
    "BINFMT".
begin_assignment_name('FS') -->
    "FS".
begin_assignment_name('OFS') -->
    "OFS".

end_clauses([end([Action])]) -->
    "END",
    ws,
    "{",
    ws,
    end_action(Action),
    ws,
    "}",
    ws,
    !.
end_clauses([]) -->
    [].

end_action(Action) -->
    for_in_action(Action),
    !.
end_action(Action) -->
    print_action(Action).

for_in_action(for_in(var(LoopVar), var(ArrayName), Body)) -->
    "for",
    ws,
    "(",
    ws,
    identifier(LoopVar),
    ws,
    "in",
    identifier_boundary,
    ws,
    identifier(ArrayName),
    ws,
    ")",
    ws,
    for_in_body(Body).

for_in_body(Actions) -->
    action_block(Actions),
    !.
for_in_body([PrintAction]) -->
    print_action(PrintAction).

actions([Action | Actions]) -->
    action(Action),
    actions_rest(Actions).

actions_rest([Action | Actions]) -->
    ws,
    ";",
    ws,
    !,
    action(Action),
    actions_rest(Actions).
actions_rest([]) -->
    [].

action(Action) -->
    if_action(Action),
    !.
action(Action) -->
    printf_action(Action),
    !.
action(Action) -->
    print_action(Action),
    !.
action(Action) -->
    next_action(Action),
    !.
action(Action) -->
    break_action(Action),
    !.
action(Action) -->
    add_assign_action(Action),
    !.
action(Action) -->
    assignment_action(Action),
    !.
action(Action) -->
    increment_action(Action),
    !.

%% if_action(-Action)//
%
%  awk conditionals: `else` is optional (an absent else parses as an
%  empty branch), and `else if` chains nest as a single-element else
%  branch containing the next if.
if_action(if(Pattern, ThenActions, ElseActions)) -->
    "if",
    ws,
    "(",
    ws,
    condition_pattern(Pattern),
    ws,
    ")",
    ws,
    action_block(ThenActions),
    if_else_part(ElseActions).

if_else_part(ElseActions) -->
    "else",
    identifier_boundary,
    if_else_body(ElseActions),
    !.
if_else_part([]) -->
    [].

if_else_body([ElseIfAction]) -->
    required_ws,
    if_action(ElseIfAction),
    !.
if_else_body(ElseActions) -->
    ws,
    action_block(ElseActions).

condition_pattern(Pattern) -->
    or_pattern(Pattern).

increment_action(inc_assoc(var(Name), KeyExpr)) -->
    identifier(Name),
    ws,
    "[",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    "]",
    !,
    "++".
increment_action(inc(var(Name))) -->
    identifier(Name),
    "++".

next_action(next) -->
    "next",
    identifier_boundary.

break_action(break) -->
    "break",
    identifier_boundary.

add_assign_action(add(var(Name), Delta)) -->
    identifier(Name),
    ws,
    "+=",
    ws,
    scalar_delta_expr(Delta).

assignment_action(set(var(Name), Value)) -->
    identifier(Name),
    ws,
    "=",
    ws,
    scalar_value_expr(Value).

scalar_value_expr(Value) -->
    scalar_delta_expr(Value).

scalar_delta_expr(Expr) -->
    i64_binary_surface_expr(Expr).
% Bare float leaves before the integer clause: "0.5" must not stop at
% the integer prefix "0".
scalar_delta_expr(Expr) -->
    float_field_expr(Expr).
scalar_delta_expr(Expr) -->
    float_literal_expr(Expr).
scalar_delta_expr(int(Value)) -->
    integer_codes(ValueCodes),
    { ValueCodes \== [],
      number_codes(Value, ValueCodes),
      Value >= 0 }.
scalar_delta_expr(special('NR')) -->
    "NR".
scalar_delta_expr(special('NF')) -->
    "NF".
scalar_delta_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
scalar_delta_expr(int(Field)) -->
    int_field_expr(int(Field)).
scalar_delta_expr(length(Field)) -->
    "length",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
scalar_delta_expr(index(Field, string(Needle))) -->
    "index",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ",",
    ws,
    quoted_string(NeedleCodes),
    ws,
    ")",
    { Field = field(_),
      NeedleCodes \== [],
      string_codes(Needle, NeedleCodes) }.
scalar_delta_expr(Expr) -->
    prolog_call_expr(Expr),
    !.
scalar_delta_expr(var(Name)) -->
    identifier(Name).

print_action(print(Fields)) -->
    "print",
    required_ws,
    print_fields(Fields).

printf_action(printf(string(Format), Args)) -->
    "printf",
    required_ws,
    quoted_string(FormatCodes),
    printf_args(Args),
    { string_codes(Format, FormatCodes) }.

printf_args([Arg | Args]) -->
    ws,
    ",",
    ws,
    !,
    field_expr(Arg),
    printf_args_rest(Args).
printf_args([]) -->
    [].

printf_args_rest([Arg | Args]) -->
    ws,
    ",",
    ws,
    !,
    field_expr(Arg),
    printf_args_rest(Args).
printf_args_rest([]) -->
    [].

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

field_expr(Expr) -->
    i64_binary_surface_expr(Expr).
field_expr(special('NR')) -->
    "NR".
field_expr(special('NF')) -->
    "NF".
field_expr(int(Field)) -->
    int_field_expr(int(Field)).
field_expr(length(Field)) -->
    "length",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
field_expr(substr(Field, Start, Len)) -->
    "substr",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ",",
    ws,
    integer_codes(StartCodes),
    ws,
    ",",
    ws,
    integer_codes(LenCodes),
    ws,
    ")",
    { Field = field(_),
      StartCodes \== [], LenCodes \== [],
      number_codes(Start, StartCodes), Start >= 1,
      number_codes(Len, LenCodes), Len >= 0 }.
field_expr(index(Field, string(Needle))) -->
    "index",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ",",
    ws,
    quoted_string(NeedleCodes),
    ws,
    ")",
    { Field = field(_),
      NeedleCodes \== [],
      string_codes(Needle, NeedleCodes) }.
field_expr(tolower(Field)) -->
    "tolower",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
field_expr(toupper(Field)) -->
    "toupper",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.
field_expr(assoc(var(Name), KeyExpr)) -->
    identifier(Name),
    ws,
    "[",
    ws,
    assoc_key_expr(KeyExpr),
    ws,
    "]",
    !.
field_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
field_expr(string(Value)) -->
    quoted_string(ValueCodes),
    { string_codes(Value, ValueCodes)
    }.
field_expr(Expr) -->
    float_field_expr(Expr),
    !.
field_expr(Expr) -->
    float_literal_expr(Expr),
    !.
field_expr(Expr) -->
    prolog_call_expr(Expr),
    !.
field_expr(var(Name)) -->
    identifier(Name).

int_field_expr(int(Field)) -->
    "int",
    ws,
    "(",
    ws,
    field_expr(Field),
    ws,
    ")",
    { Field = field(_) }.

%% i64_binary_surface_expr(-Expr)//
%
%  General native i64 arithmetic with awk precedence: * / % bind tighter
%  than + -, both levels associate left, and parentheses group. Factors
%  are the native i64 primaries plus integer literals and bare numeric
%  field coercions such as `$3` (zero when the field is not a strict
%  signed decimal). The top-level result must contain at least one
%  operator so bare primaries keep their existing print/slice meaning.
i64_binary_surface_expr(Expr) -->
    i64_additive_expr(Expr),
    { i64_binary_expr_ast(Expr) }.

i64_binary_expr_ast(Expr) :-
    compound(Expr),
    functor(Expr, Functor, 2),
    memberchk(Functor, [add_i64, sub_i64, mul_i64, div_i64, mod_i64]).

i64_additive_expr(Expr) -->
    i64_multiplicative_expr(First),
    i64_additive_chain(First, Expr).

i64_additive_chain(Acc, Expr) -->
    ws,
    i64_additive_operator(Functor),
    ws,
    i64_multiplicative_expr(Right),
    !,
    { Acc1 =.. [Functor, Acc, Right] },
    i64_additive_chain(Acc1, Expr).
i64_additive_chain(Expr, Expr) -->
    [].

i64_multiplicative_expr(Expr) -->
    i64_factor_expr(First),
    i64_multiplicative_chain(First, Expr).

i64_multiplicative_chain(Acc, Expr) -->
    ws,
    i64_multiplicative_operator(Functor),
    ws,
    i64_factor_expr(Right),
    !,
    { Acc1 =.. [Functor, Acc, Right] },
    i64_multiplicative_chain(Acc1, Expr).
i64_multiplicative_chain(Expr, Expr) -->
    [].

i64_additive_operator(add_i64) -->
    "+".
i64_additive_operator(sub_i64) -->
    "-".

i64_multiplicative_operator(mul_i64) -->
    "*".
i64_multiplicative_operator(div_i64) -->
    "/".
i64_multiplicative_operator(mod_i64) -->
    "%".

i64_factor_expr(Expr) -->
    "(",
    ws,
    i64_additive_expr(Expr),
    ws,
    ")",
    !.
i64_factor_expr(Expr) -->
    i64_binary_primary_expr(Expr),
    !.
i64_factor_expr(Expr) -->
    float_field_expr(Expr),
    !.
i64_factor_expr(Expr) -->
    float_literal_expr(Expr),
    !.
i64_factor_expr(int(Value)) -->
    integer_codes(ValueCodes),
    !,
    { ValueCodes \== [],
      number_codes(Value, ValueCodes)
    }.
i64_factor_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    !,
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
i64_factor_expr(Expr) -->
    prolog_call_expr(Expr),
    !.
i64_factor_expr(var(Name)) -->
    identifier(Name).

%% prolog_call_expr(-Expr)//
%
%  A named Prolog predicate as an i64 expression: `pred(args...)` calls
%  the compiled predicate with one extra trailing output argument and
%  yields its integer binding, or 0 when the call fails or binds a
%  non-integer.
prolog_call_expr(prolog_call(Name, Args)) -->
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")".

%% float_literal_expr(-Expr)//
%
%  A decimal float literal such as 2.5 or 0.1, kept exact as
%  float_const(Mantissa, Denominator) with Denominator = 10^k so
%  codegen can emit a correctly rounded double without a lossy
%  Prolog-float round trip (LLVM rejects inexact decimal FP text).
float_literal_expr(float_const(Mantissa, Denominator)) -->
    integer_codes(IntCodes),
    ".",
    integer_codes(FracCodes),
    { IntCodes \== [],
      FracCodes \== [],
      append(IntCodes, FracCodes, AllCodes),
      number_codes(Mantissa, AllCodes),
      length(FracCodes, FracLen),
      Denominator is 10 ** FracLen
    }.

%% float_field_expr(-Expr)//
%
%  awk-style numeric coercion to double: float($N) parses the field
%  with strtod semantics (leading number, trailing text ignored, 0.0
%  when non-numeric).
float_field_expr(float_field(Index)) -->
    "float",
    ws,
    "(",
    ws,
    "$",
    integer_codes(IndexCodes),
    ws,
    ")",
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.

i64_binary_primary_expr(special('NR')) -->
    "NR".
i64_binary_primary_expr(special('NF')) -->
    "NF".
i64_binary_primary_expr(Expr) -->
    int_field_expr(Expr).
i64_binary_primary_expr(length(Field)) -->
    "length",
    ws,
    "(",
    ws,
    simple_field_expr(Field),
    ws,
    ")".
i64_binary_primary_expr(index(Field, string(Needle))) -->
    "index",
    ws,
    "(",
    ws,
    simple_field_expr(Field),
    ws,
    ",",
    ws,
    quoted_string(NeedleCodes),
    ws,
    ")",
    { NeedleCodes \== [],
      string_codes(Needle, NeedleCodes) }.

simple_field_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.

assoc_key_expr(field(Index)) -->
    "$",
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    }.
assoc_key_expr(int(Value)) -->
    signed_integer_value(Value),
    !.
assoc_key_expr(string(Value)) -->
    quoted_string(ValueCodes),
    { ValueCodes \== [],
      string_codes(Value, ValueCodes)
    }.
assoc_key_expr(var(Name)) -->
    identifier(Name).

identifier(Name) -->
    identifier_start(Start),
    identifier_rest(Rest),
    { atom_codes(Name, [Start | Rest]) }.

identifier_start(Code) -->
    [Code],
    { code_type(Code, alpha) -> true ; Code =:= 0'_ }.

identifier_rest([Code | Codes]) -->
    [Code],
    { code_type(Code, alnum) -> true ; Code =:= 0'_ },
    !,
    identifier_rest(Codes).
identifier_rest([]) -->
    [].

identifier_boundary([Code | Rest], [Code | Rest]) :-
    \+ identifier_continue_code(Code).
identifier_boundary([], []).

identifier_continue_code(Code) :-
    code_type(Code, alnum),
    !.
identifier_continue_code(0'_).

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
