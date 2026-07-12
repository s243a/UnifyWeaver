% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_parser, [
    plawk_parse_string/2,
    plawk_parse_source/3
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
    plawk_parse_source(Source, Program, _PrologClauses).

%% plawk_parse_source(+Source, -Program, -PrologClauses) is semidet.
%
%  Like plawk_parse_string/2, but also lifts embedded Prolog blocks:
%
%      @prolog
%      weight(I, F, R) :- R is I * F.
%      @end
%
%  Block markers sit alone on their line (leading/trailing blanks ok).
%  A heredoc-style tag makes the fence unambiguous when the Prolog
%  text itself contains an @end-shaped line: `@prolog-TAG` closes only
%  at `@end-TAG` with the exact same tag. Blocks may appear anywhere
%  between top-level program parts; their text never routes through
%  the awk grammar -- it is term-read as ordinary Prolog and returned
%  as PrologClauses for the compile driver to hand to
%  write_wam_llvm_project alongside the program.
plawk_parse_source(Source, Program, PrologClauses) :-
    string(Source),
    plawk_split_prolog_blocks(Source, Stripped, BlockTexts),
    maplist(plawk_read_block_clauses, BlockTexts, ClausesNested),
    append(ClausesNested, BlockClauses),
    string_codes(Stripped, Codes),
    phrase(plawk_program(Program, FunctionClauses, DynEntries), Codes),
    append(BlockClauses, FunctionClauses, PrologClauses),
    plawk_dynentry_reserved_check(DynEntries, PrologClauses).

%% plawk_dynentry_reserved_check(+DynEntries, +PrologClauses)
%
%  DYNENTRY reserves a name for the compiled object -- declaring one
%  that an @prolog block or `function` definition also defines, or a
%  surface builtin/keyword name, throws (a parse error at the CLI):
%  never silent shadowing in either direction.
plawk_dynentry_reserved_check([], _) :- !.
plawk_dynentry_reserved_check(DynEntries, PrologClauses) :-
    findall(Defined,
        ( member(Clause, PrologClauses),
          plawk_clause_head_name(Clause, Defined)
        ),
        DefinedNames),
    forall(member(Name, DynEntries),
        (   memberchk(Name, DefinedNames)
        ->  throw(error(plawk_dynentry_reserved(Name),
                context(plawk_parse_source/3,
                    'DYNENTRY name is also defined by an @prolog block or function -- reservation forbids shadowing')))
        ;   plawk_surface_reserved_name(Name)
        ->  throw(error(plawk_dynentry_reserved(Name),
                context(plawk_parse_source/3,
                    'DYNENTRY name collides with a surface builtin/keyword')))
        ;   true
        )).

plawk_clause_head_name((Head :- _Body), Name) :-
    !,
    functor(Head, Name, _).
plawk_clause_head_name(Head, Name) :-
    compound(Head),
    functor(Head, Name, _).

plawk_surface_reserved_name(Name) :-
    memberchk(Name, [length, index, substr, int, tolower, toupper,
        float, blob, dyncall, dyncall_at, compile, compile_file,
        print, printf, function, for, in, if, else, next, break]).

plawk_split_prolog_blocks(Source, Stripped, BlockTexts) :-
    split_string(Source, "\n", "", Lines),
    plawk_split_block_lines(Lines, KeptLines, BlockTexts),
    atomic_list_concat(KeptLines, '\n', StrippedAtom),
    atom_string(StrippedAtom, Stripped).

plawk_split_block_lines([], [], []).
plawk_split_block_lines([Line | Lines], KeptLines, [BlockText | BlockTexts]) :-
    plawk_prolog_open_marker(Line, EndMarker),
    !,
    plawk_take_block_lines(Lines, EndMarker, BlockLines, Rest),
    atomic_list_concat(BlockLines, '\n', BlockAtom),
    atom_string(BlockAtom, BlockText),
    plawk_split_block_lines(Rest, KeptLines, BlockTexts).
plawk_split_block_lines([Line | Lines], [Line | KeptLines], BlockTexts) :-
    plawk_split_block_lines(Lines, KeptLines, BlockTexts).

plawk_prolog_open_marker(Line, EndMarker) :-
    split_string(Line, "", " \t", [Trimmed]),
    ( Trimmed == "@prolog"
    ->  EndMarker = "@end"
    ;   string_concat("@prolog-", Tag, Trimmed),
        Tag \== "",
        string_concat("@end-", Tag, EndMarker)
    ).

% an unterminated block fails the parse (no clause for [])
plawk_take_block_lines([Line | Lines], EndMarker, BlockLines, Rest) :-
    split_string(Line, "", " \t", [Trimmed]),
    ( Trimmed == EndMarker
    ->  BlockLines = [],
        Rest = Lines
    ;   BlockLines = [Line | BlockLines1],
        plawk_take_block_lines(Lines, EndMarker, BlockLines1, Rest)
    ).

plawk_read_block_clauses(BlockText, Clauses) :-
    setup_call_cleanup(
        open_string(BlockText, Stream),
        plawk_read_stream_clauses(Stream, Clauses),
        close(Stream)).

plawk_read_stream_clauses(Stream, Clauses) :-
    read_term(Stream, Term, []),
    ( Term == end_of_file
    ->  Clauses = []
    ;   Clauses = [Term | Rest],
        plawk_read_stream_clauses(Stream, Rest)
    ).

plawk_program(Program) -->
    plawk_program(Program, _FunctionClauses).

plawk_program(Program, FunctionClauses) -->
    plawk_program(Program, FunctionClauses, _DynEntries).

plawk_program(program(BeginClauses, Rules, EndClauses), FunctionClauses,
        DynEntries) -->
    ws,
    begin_clauses(BeginClauses),
    function_defs(FunctionClauses),
    dynentry_decls(DynEntries),
    program_rules(Rules0),
    end_clauses(EndClauses0),
    eos,
    { plawk_dynentry_rewrite_all(Rules0, DynEntries, Rules),
      plawk_dynentry_rewrite_all(EndClauses0, DynEntries, EndClauses)
    }.

%% dynentry_decls(-Names)//
%
%  Surface B (declaration-bound library names): for a fixed DYNLOAD
%  shipping a known entry family, bind names once at declaration and
%  call them like ordinary functions:
%
%      DYNENTRY parse, score
%      { total += parse($1) + score($2) }
%
%  A declared name is RESERVED for the compiled object: every bare
%  `name(args)` call site (and `float(name(args))`) rewrites at parse
%  time to the named-entry node (dyncall_named / float_dyncall_named),
%  so resolution stays static -- the entry's PC resolves once at
%  startup and is cached, exactly like an explicit dyncall@name site.
%  Reservation makes the name sets disjoint by construction: declaring
%  a name that an @prolog block or `function` definition also defines,
%  or a surface builtin name, is a PARSE ERROR (never silent
%  shadowing) -- checked in plawk_parse_source once the block clauses
%  are known.
dynentry_decls(Names) -->
    dynentry_decl(First),
    !,
    dynentry_decls(Rest),
    { append(First, Rest, Names) }.
dynentry_decls([]) -->
    [].

dynentry_decl(Names) -->
    "DYNENTRY",
    identifier_boundary,
    ws,
    dynentry_names(Names),
    ws.

dynentry_names([Name | Names]) -->
    identifier(Name),
    dynentry_names_rest(Names).

dynentry_names_rest([Name | Names]) -->
    ws,
    ",",
    ws,
    !,
    identifier(Name),
    dynentry_names_rest(Names).
dynentry_names_rest([]) -->
    [].

%% plawk_dynentry_rewrite_all(+Terms0, +Names, -Terms)
%
%  Substitute bare-call nodes over declared names with their
%  named-entry counterparts, everywhere in the parsed rules/END:
%  prolog_call -> dyncall_named, float_call -> float_dyncall_named.
%  A declared name in a position the named machinery does not compile
%  (e.g. a guard pattern) yields the ordinary
%  outside-the-compilable-surface build error rather than silently
%  calling userspace.
plawk_dynentry_rewrite_all(Terms, [], Terms) :- !.
plawk_dynentry_rewrite_all(Terms0, Names, Terms) :-
    plawk_dynentry_rewrite(Terms0, Names, Terms).

plawk_dynentry_rewrite(Term0, Names, Term) :-
    (   Term0 = prolog_call(Name, Args0),
        memberchk(Name, Names)
    ->  plawk_dynentry_rewrite(Args0, Names, Args),
        Term = dyncall_named(Name, Args)
    ;   Term0 = float_call(Name, Args0),
        memberchk(Name, Names)
    ->  plawk_dynentry_rewrite(Args0, Names, Args),
        Term = float_dyncall_named(Name, Args)
    ;   compound(Term0)
    ->  Term0 =.. [F | Args0],
        maplist(plawk_dynentry_rewrite_arg(Names), Args0, Args),
        Term =.. [F | Args]
    ;   Term = Term0
    ).

plawk_dynentry_rewrite_arg(Names, Arg0, Arg) :-
    plawk_dynentry_rewrite(Arg0, Names, Arg).

%% function_defs(-Clauses)//
%
%  awk-style expression functions, pure sugar over the foreign bridge:
%
%      function scale(a, b) { return a * b + 1 }
%
%  desugars at parse time into the Prolog clause
%
%      scale(A, B, R) :- R is A * B + 1.
%
%  and is called like any bridged predicate: `scale($1, $2)` as an
%  integer expression, `float(scale($1, $2))` to keep fractions. The
%  body is one `return` of an arithmetic expression over the
%  parameters (awk precedence; % maps to Prolog mod); an identifier
%  that is not a parameter fails the parse.
function_defs([Clause | Clauses]) -->
    function_def(Clause),
    !,
    function_defs(Clauses).
function_defs([]) -->
    [].

function_def((Head :- Body)) -->
    "function",
    identifier_boundary,
    ws,
    identifier(Name),
    ws,
    "(",
    ws,
    function_params(Params),
    ws,
    ")",
    ws,
    "{",
    ws,
    "return",
    identifier_boundary,
    ws,
    function_expr(Params, Pairs, ArithTerm),
    action_block_close,
    { pairs_values(Pairs, Vars),
      append(Vars, [Result], HeadArgs),
      Head =.. [Name | HeadArgs],
      Body = (Result is ArithTerm)
    }.

function_params([Param | Params]) -->
    identifier(Param),
    function_params_rest(Params).

function_params_rest([Param | Params]) -->
    ws,
    ",",
    ws,
    !,
    identifier(Param),
    function_params_rest(Params).
function_params_rest([]) -->
    [].

% arithmetic over the parameters, with awk precedence: * / % bind
% tighter than + -, both associate left, parentheses group
function_expr(Params, Pairs, Term) -->
    { pairs_keys(Pairs, Params) },
    function_additive(Pairs, Term).

function_additive(Pairs, Term) -->
    function_multiplicative(Pairs, First),
    function_additive_chain(Pairs, First, Term).

function_additive_chain(Pairs, Acc, Term) -->
    ws,
    function_add_op(Op),
    ws,
    !,
    function_multiplicative(Pairs, Right),
    { Acc1 =.. [Op, Acc, Right] },
    function_additive_chain(Pairs, Acc1, Term).
function_additive_chain(_Pairs, Term, Term) -->
    [].

function_add_op(+) --> "+".
function_add_op(-) --> "-".

function_multiplicative(Pairs, Term) -->
    function_factor(Pairs, First),
    function_multiplicative_chain(Pairs, First, Term).

function_multiplicative_chain(Pairs, Acc, Term) -->
    ws,
    function_mul_op(Op),
    ws,
    !,
    function_factor(Pairs, Right),
    { Acc1 =.. [Op, Acc, Right] },
    function_multiplicative_chain(Pairs, Acc1, Term).
function_multiplicative_chain(_Pairs, Term, Term) -->
    [].

function_mul_op(*) --> "*".
function_mul_op(/) --> "/".
function_mul_op(mod) --> "%".

function_factor(Pairs, Term) -->
    "(",
    ws,
    !,
    function_additive(Pairs, Term),
    ws,
    ")".
function_factor(_Pairs, Value) -->
    float_literal_expr(float_const(Mantissa, Denominator)),
    !,
    { Value is Mantissa / Denominator }.
function_factor(_Pairs, Value) -->
    integer_codes(ValueCodes),
    { ValueCodes \== [] },
    !,
    { number_codes(Value, ValueCodes) }.
function_factor(Pairs, Var) -->
    identifier(Param),
    { memberchk(Param-Var, Pairs) }.

% Tagged-union programs route rules through per-arm case blocks: the
% arm index scopes the field types of every rule inside the block.
program_rules(case_blocks(Blocks)) -->
    case_block(Block),
    !,
    case_blocks_rest(Blocks0),
    { Blocks = [Block | Blocks0] }.
program_rules(Rules) -->
    rules(Rules).

case_blocks_rest([Block | Blocks]) -->
    ws,
    case_block(Block),
    !,
    case_blocks_rest(Blocks).
case_blocks_rest([]) -->
    ws.

case_block(case_arm(Index, Rules)) -->
    "case",
    identifier_boundary,
    ws,
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    },
    ws,
    "{",
    ws,
    rules(Rules),
    ws,
    "}".

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
    action_block_close.

% a trailing `;` or newline before the closing brace is harmless
action_block_close -->
    action_sep,
    !,
    "}",
    ws.
action_block_close -->
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
    tag_eq_pattern(Pattern),
    !.
% blob(dyncall...) == "literal" -- an equality guard on a runtime
% grammar's byte output (JIT roadmap item 2 follow-on). Must precede
% prolog_guard_pattern: blob(...) would otherwise parse as a foreign
% guard call named blob.
base_pattern(Pattern) -->
    blob_eq_pattern(Pattern),
    !.
base_pattern(Pattern) -->
    prolog_guard_pattern(Pattern).

%% blob_eq_pattern(-Pattern)//
blob_eq_pattern(blob_eq(Blob, Value)) -->
    blob_call_expr(Blob),
    ws,
    "==",
    ws,
    quoted_string(ValueCodes),
    { ValueCodes \== [],
      string_codes(Value, ValueCodes)
    }.

%% tag_eq_pattern(-Pattern)//
%
%  TAG == K guards a rule by the record tag of a tagged-union layout.
%  Surface sugar: the codegen groups TAG-guarded rules into the same
%  per-arm case blocks that `case K { ... }` produces, so the tag test
%  must be the leftmost conjunct of the rule's pattern.
tag_eq_pattern(tag_pat(Tag)) -->
    "TAG",
    identifier_boundary,
    ws,
    "==",
    ws,
    integer_codes(TagCodes),
    { TagCodes \== [],
      number_codes(Tag, TagCodes),
      Tag >= 0
    }.

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

%% dyncall_at_source(-Source)//
%
%  The first argument of dyncall_at(...): either a path (field / string
%  literal, as before), or compile(field-or-string) -- the eval surface.
%  compile(Src) runs the shipped bootstrap-compiler .wamo on the Prolog
%  source text Src at runtime and yields a HANDLE to the freshly
%  compiled grammar (deduplicated by source text, so repeated compiles
%  of the same source reuse one loaded grammar). "compile" is reserved
%  in this position only; elsewhere it still parses as an ordinary
%  identifier.
% compile_file(field-or-string): the path names a grammar SOURCE file
% read at runtime; its contents compile through the same handle
% registry as compile(...), so editing the file changes behaviour with
% no rebuild (content dedup recompiles exactly when the bytes differ).
% Parsed before compile: they share a prefix.
dyncall_at_source(compile_file_src(Arg)) -->
    "compile_file",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
dyncall_at_source(compile_src(Arg)) -->
    "compile",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
% a bare identifier: a grammar HANDLE stored in a scalar --
%
%     { h = compile("[(sq(...)...)]") ; total += dyncall_at@sq(h, $1) }
%
% names the compiled source once instead of repeating it per call
% site. The handle IS an i64 (a registry index), so it rides the
% scalar machinery; the source marshals as (null path, handle id), the
% registry discriminator @plawk_dyncall_at_get already speaks.
dyncall_at_source(handle_src(var(Name))) -->
    identifier(Name),
    !.
dyncall_at_source(Source) -->
    foreign_arg(Source).

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
begin_assignment_name('OUTFMT') -->
    "OUTFMT".
begin_assignment_name('DYNLOAD') -->
    "DYNLOAD".
begin_assignment_name('DYNCACHE') -->
    "DYNCACHE".
begin_assignment_name('FS') -->
    "FS".
begin_assignment_name('OFS') -->
    "OFS".

% END accumulate-then-print (assoc for-in, stage 2): a for-in that folds
% the hash into a scalar, followed by a print that reads the accumulator.
% `END { for (k in arr) acc += arr[k] ; print acc }`. The for-in body is a
% single `acc += OPERAND` where OPERAND is the iterated value `arr[k]`, the
% loop key `k`, or an integer; the trailing print reads `acc` (the
% loop-carried total). Tried before the single-action END clause -- the
% two-statement shape (for-in `;` print) is unambiguous. See
% PLAWK_ASSOC_FORIN.md.
end_clauses([end([for_in(var(LoopVar), var(ArrayName), [AccAction]),
                  print(PrintFields)])]) -->
    "END", ws, "{", ws,
    "for", ws, "(", ws,
    identifier(LoopVar), ws, "in", identifier_boundary, ws,
    identifier(ArrayName), ws, ")", ws,
    forin_accum_action(ArrayName, LoopVar, AccAction), ws,
    action_sep, ws,
    print_action(print(PrintFields)), ws,
    "}", ws,
    !.
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

% `acc += OPERAND` inside a for-in accumulate body. The operand is
% for-in-scoped: `arr[k]` (the iterated value; array/key must match the
% loop), `k` (the loop key), or an integer literal.
forin_accum_action(Array, Key, add(var(Acc), Operand)) -->
    identifier(Acc), ws, "+=", ws,
    forin_accum_operand(Array, Key, Operand).

forin_accum_operand(Array, Key, forin_val(Array)) -->
    identifier(Array), ws, "[", ws, identifier(Key), ws, "]",
    !.
forin_accum_operand(_Array, Key, forin_key) -->
    identifier(Key),
    !.
forin_accum_operand(_Array, _Key, int(Value)) -->
    signed_integer_value(Value).

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
% for-in filter (assoc for-in, stage 1): `{ if (GUARD) print ... }` where
% GUARD compares the loop key `k` or the value `arr[k]` to an integer.
% A for-in-scoped condition -- k / arr[k] are only meaningful inside the
% loop, so the operands live here rather than in the global pattern
% grammar. Gates the per-key print; no cross-iteration state.
for_in_body([if(Guard, [PrintAction], [])]) -->
    "{", ws, "if", ws, "(", ws, forin_guard(Guard), ws, ")", ws,
    print_action(PrintAction), ws, "}",
    !.
for_in_body([WritebinAction]) -->
    writebin_action(WritebinAction),
    !.
for_in_body([PrintAction]) -->
    print_action(PrintAction).

% arr[k] CMP int -- value comparison (tried before the bare-key form,
% since `arr[` also starts with an identifier).
forin_guard(forin_val_cmp(Array, LoopVar, Op, Value)) -->
    identifier(Array), ws, "[", ws, identifier(LoopVar), ws, "]", ws,
    numeric_cmp_op(Op), ws, signed_integer_value(Value),
    !.
% k CMP int -- loop-key comparison.
forin_guard(forin_key_cmp(LoopVar, Op, Value)) -->
    identifier(LoopVar), ws, numeric_cmp_op(Op), ws,
    signed_integer_value(Value).

actions([Action | Actions]) -->
    action(Action),
    actions_rest(Action, Actions).

actions_rest(_Prev, [Action | Actions]) -->
    action_sep,
    action(Action),
    !,
    actions_rest(Action, Actions).
% as in awk/C, no separator is needed after a compound statement's
% closing brace (whose trailing ws has already been consumed)
actions_rest(Prev, [Action | Actions]) -->
    { plawk_block_action(Prev) },
    action(Action),
    !,
    actions_rest(Action, Actions).
actions_rest(_Prev, []) -->
    [].

plawk_block_action(if(_Pattern, _Then, _Else)).
plawk_block_action(foreach(_Body)).

%% action_sep//0
%
%  One statement separator, as in awk: any run of blanks, comments,
%  semicolons, and newlines containing at least one `;` or newline.
%  (A trailing separator before `}` is harmless: the following
%  action// fails and actions_rest backtracks to its empty clause.)
action_sep -->
    action_sep_scan(no).

action_sep_scan(_Seen) -->
    ";",
    !,
    action_sep_scan(yes).
action_sep_scan(_Seen) -->
    "\n",
    !,
    action_sep_scan(yes).
action_sep_scan(Seen) -->
    [Code],
    { Code =\= 0'\n, code_type(Code, space) },
    !,
    action_sep_scan(Seen).
action_sep_scan(Seen) -->
    "#",
    !,
    comment_rest,
    action_sep_scan(Seen).
action_sep_scan(yes) -->
    [].

action(Action) -->
    if_action(Action),
    !.
action(Action) -->
    printf_action(Action),
    !.
action(Action) -->
    writebin_action(Action),
    !.
action(Action) -->
    foreach_action(Action),
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
    dynrec_view_action(Action),
    !.
action(Action) -->
    dynassoc_bind_action(Action),
    !.
action(Action) -->
    dynposarray_bind_action(Action),
    !.
action(Action) -->
    dynrec_bind_action(Action),
    !.
% for (k in arr) { ... } as a RULE-BODY action: per-record iteration
% over an assoc table (e.g. one a grammar just populated via
% `arr = dyncall@t($1) as assoc`), not just the END report. The body
% grammar is shared with the END form; what compiles is gated by the
% assoc-route codegen (a print body over the loop key / lookups).
action(Action) -->
    for_in_action(Action),
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

%% dynrec_bind_action(-Action)//
%
%  Structured-return destructuring: bind each field of a grammar's compound
%  return to a scalar variable, typed by a return shape.
%
%      (n, half) = dyncall@rec($1) as (i64 f64)
%
%  desugars to dynrec_bind([n, half], dyncall_named(rec, [field(1)]),
%  [i64, f64]); field 0 lands in the i64 scalar n, field 1 in the f64
%  scalar half. The call is a bare dyncall(...) (default entry) or a
%  dyncall@name(...) (named entry); the type list is whitespace-separated
%  i64 / f64 tokens (one per bound variable).
%% dynrec_view_action(-Action)//
%
%  Structured-return record view: the returned compound becomes the current
%  record for a scoped block, so `$1`,`$2` read its fields like a BINFMT
%  line.
%
%      dyncall@rec($1) as (i64 f64) { total += $1 ; sum += $2 }
%
%  desugars (in codegen) to a destructure into hidden temporaries plus the
%  block body with `$N` rewritten to the Nth temporary -- so it rides the
%  same machinery as the explicit destructure, no field-pointer repoint.
%% dynassoc_bind_action(-Action)//
%
%  Associative-array return: a grammar returning a list of integer key-value
%  pairs populates a plawk assoc array.
%
%      arr = dyncall@tally($1) as assoc
%      arr = dyncall($1) as assoc          % DYNLOAD object's default entry
%      arr = dyncall@label($1) as assoc(str)   % string VALUES
%
%  desugars to dynassoc_bind(var(arr), dyncall_named(tally, [field(1)]))
%  (or dynassoc_bind(var(arr), dyncall([field(1)])) for the default-entry
%  form); per record the returned [K-V, ...] pairs are inserted into arr's
%  i64 table, so END `arr[key]` lookups see the accumulated result.
%
%  The `(str)` value kind yields dynassoc_bind_str(var(arr), Call): the
%  grammar returns ATOM values, stored by registry id with replace (not
%  accumulate) semantics, and reads print the text.
dynassoc_bind_action(Action) -->
    identifier(Name),
    ws,
    "=",
    ws,
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "assoc",
    dynassoc_value_kind(Kind),
    { Kind == str
    ->  Action = dynassoc_bind_str(var(Name), Call)
    ;   Action = dynassoc_bind(var(Name), Call)
    }.

dynassoc_value_kind(str) -->
    ws, "(", ws, "str", ws, ")",
    !.
dynassoc_value_kind(i64) -->
    [].

%% dynposarray_bind_action(-Action)//
%
%  Positional-array return: a grammar returning a FLAT list of integers
%  populates a plawk array by POSITION -- element i lands at key i
%  (1-indexed, the awk `split` convention), replace semantics (the array
%  reflects the most recent record's list).
%
%      arr = dyncall@fields($1) as array
%      arr = dyncall($1) as array          % DYNLOAD object's default entry
%
%  desugars to dynposarray_bind(var(arr), dyncall_named(fields, [field(1)]))
%  (or dynposarray_bind(var(arr), dyncall([field(1)])) for the default
%  entry). The last target container of JIT roadmap item 4: same i64
%  table as `as assoc`, but filled from a flat [V1, ..., Vn] list instead
%  of key-value pairs, so END `arr[1]`, `arr[2]`, ... and for-in read the
%  positional values.
dynposarray_bind_action(Action) -->
    identifier(Name),
    ws,
    "=",
    ws,
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "array",
    dynassoc_value_kind(Kind),
    { Kind == str
    ->  Action = dynposarray_bind_str(var(Name), Call)
    ;   Action = dynposarray_bind(var(Name), Call)
    }.

dynrec_view_action(dynrec_view(Call, Types, Body)) -->
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "(",
    ws,
    dynrec_type_list(Types),
    ws,
    ")",
    ws,
    action_block(Body).

dynrec_bind_action(dynrec_bind(Vars, Call, Types)) -->
    "(",
    ws,
    dynrec_var_list(Vars),
    ws,
    ")",
    ws,
    "=",
    ws,
    dynrec_call_expr(Call),
    ws,
    "as",
    identifier_boundary,
    ws,
    "(",
    ws,
    dynrec_type_list(Types),
    ws,
    ")".

dynrec_var_list([Name | Rest]) -->
    identifier(Name),
    dynrec_var_list_rest(Rest).
dynrec_var_list_rest([Name | Rest]) -->
    ws, ",", ws, identifier(Name), !, dynrec_var_list_rest(Rest).
dynrec_var_list_rest([]) -->
    [].

dynrec_call_expr(Call) -->
    prolog_call_expr(Call),
    { ( Call = dyncall(_)
      ; Call = dyncall_named(_, _)
      ; Call = dyncall_at(_, _)                % runtime source (JIT reader)
      ; Call = dyncall_at_named(_, _, _)
      ) }.

dynrec_type_list([Type | Rest]) -->
    dynrec_type(Type),
    dynrec_type_list_rest(Rest).
dynrec_type_list_rest([Type | Rest]) -->
    ws, dynrec_type(Type), !, dynrec_type_list_rest(Rest).
dynrec_type_list_rest([]) -->
    [].

dynrec_type(i64) --> "i64".
dynrec_type(f64) --> "f64".
dynrec_type(string) --> "string".

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
    float_call_expr(Expr).
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

%% foreach_action(-Action)//
%
%  foreach { actions } - run the block once per repetition element of
%  the current record; inside, $1..$M are the element's fields.
foreach_action(foreach(Actions)) -->
    "foreach",
    identifier_boundary,
    ws,
    action_block(Actions).

%% writebin_action(-Action)//
%
%  writebin expr, expr, ... - emit one fixed-layout binary record on
%  stdout, laid out per BEGIN { OUTFMT = "..." }. With a tagged-union
%  OUTFMT (`OUTFMT = "case(arm0 | arm1)"`), each site statically
%  targets one arm: `writebin case K, expr, ...` emits the 8-byte tag
%  K then arm K's slots.
writebin_action(writebin_arm(Index, Fields)) -->
    "writebin",
    required_ws,
    "case",
    identifier_boundary,
    ws,
    integer_codes(IndexCodes),
    { IndexCodes \== [],
      number_codes(Index, IndexCodes),
      Index >= 0
    },
    ws,
    ",",
    !,
    ws,
    print_fields(Fields).
writebin_action(writebin(Fields)) -->
    "writebin",
    required_ws,
    print_fields(Fields).

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
    float_call_expr(Expr),
    !.
field_expr(Expr) -->
    blob_call_expr(Expr),
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
% dyncall(args...) is reserved: it routes to a runtime-loaded .wamo
% object's entry (BEGIN { DYNLOAD = "file.wamo" }) rather than a
% compiled predicate, so it parses to its own node and never touches the
% compiled-foreign-call machinery. The cut fires only after a full
% `dyncall(...)`, so an identifier like `dyncalls(...)` still falls
% through to the generic prolog call below.
% dyncall_at(Source, args...) is the dynamic-source form: Source (a field
% or string literal) names the .wamo object at runtime, chosen per call,
% and args... are the entry's inputs. Reserved like dyncall; parsed before
% it so the longer keyword wins.
% The source may also be compile(field-or-string): compile the Prolog
% source text through the shipped bootstrap-compiler object at runtime
% (the eval surface -- JIT roadmap item 5 payoff) and dyncall the
% freshly compiled grammar. compile() dedups by source text, so a
% per-record `dyncall_at(compile($1), $2)` compiles each distinct
% grammar once and reuses it from the cache thereafter.
% dyncall_at@name(Source, args...) selects a NAMED entry of the
% runtime-chosen object -- the composition of the two selection axes:
% the source is runtime data (a path expression or a compile() handle),
% the entry name is a compile-time token. Because the object is not
% fixed, the name resolves per call against the loaded VM's
% materialized entry table (@wam_object_vm_entry_pc) rather than a
% startup-cached PC. Parsed before the bare dyncall_at so the @ form
% wins.
% compile(src) / compile_file(path) as EXPRESSIONS: yield the grammar
% HANDLE (an i64 registry index; 0 on failure) for storing in a scalar.
% Content dedup makes a per-record re-assignment a registry hit, so
% `h = compile("...")` in a rule body costs one compile per distinct
% source. Reserved like dyncall; `compilex(...)` still parses as an
% ordinary identifier call.
prolog_call_expr(compile_file_handle(Arg)) -->
    "compile_file",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
prolog_call_expr(compile_handle(Arg)) -->
    "compile",
    ws,
    "(",
    ws,
    foreign_arg(Arg),
    ws,
    ")",
    !.
prolog_call_expr(dyncall_at_named(Name, Source, Args)) -->
    "dyncall_at@",
    identifier(Name),
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    !.
prolog_call_expr(dyncall_at(Source, Args)) -->
    "dyncall_at",
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    !.
% dyncall@name(args...) selects a named entry of the runtime object (a
% multi-entry .wamo, one file exposing several predicates via
% wamo_entries([...])). The @name is a compile-time token -- the entry
% name is fixed in the source, so the shim resolves its label index once
% at startup (via @wam_object_entry_index) and caches the PC. Parsed
% before the bare `dyncall(...)` so the `@` form wins; a plain
% `dyncall(...)` still falls through to the clause below.
prolog_call_expr(dyncall_named(Name, Args)) -->
    "dyncall@",
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    !.
prolog_call_expr(dyncall(Args)) -->
    "dyncall",
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    !.
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

%% float_call_expr(-Expr)//
%
%  float(name(args)): a compiled-Prolog call whose output argument is
%  numeric and lands in a double context -- Float results keep their
%  fraction (an i64-context call would truncate the surface to
%  integers). The float(...) wrapper is what selects the
%  double-returning wrapper at codegen time.
%
%  float(dyncall(...)) / float(dyncall_at(...)) are the runtime-object
%  variants: the loaded grammar's numeric output is read as a double.
%  These must precede the generic clause, whose inner prolog_call_expr
%  would otherwise parse `dyncall` as an ordinary identifier call.
% float(dyncall_at@name(Source, args...)): a named entry on a runtime
% source, read as a double -- parsed before the bare at form so the @
% form wins.
float_call_expr(float_dyncall_at_named(Name, Source, Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall_at@",
    identifier(Name),
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_dyncall_at(Source, Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall_at",
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_dyncall_named(Name, Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall@",
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_dyncall(Args)) -->
    "float",
    ws,
    "(",
    ws,
    "dyncall",
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.
float_call_expr(float_call(Name, Args)) -->
    "float",
    ws,
    "(",
    ws,
    prolog_call_expr(prolog_call(Name, Args)),
    ws,
    ")".

%% blob_call_expr(-Expr)//
%
%  blob(dyncall(...)) / blob(dyncall_at(...)): read the runtime grammar's
%  Atom output as opaque bytes (a slice), for print / writebin positions.
%  Reserved like the dyncall forms; the inner keyword disambiguates.
% blob(dyncall_at@name(Source, args...)): a named entry on a runtime
% source, read as opaque bytes -- parsed before the bare at form.
blob_call_expr(blob_dyncall_at_named(Name, Source, Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall_at@",
    identifier(Name),
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
blob_call_expr(blob_dyncall_at(Source, Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall_at",
    ws,
    "(",
    ws,
    dyncall_at_source(Source),
    foreign_args_rest(Args),
    ws,
    ")",
    ws,
    ")",
    !.
blob_call_expr(blob_dyncall_named(Name, Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall@",
    identifier(Name),
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.
blob_call_expr(blob_dyncall(Args)) -->
    "blob",
    ws,
    "(",
    ws,
    "dyncall",
    ws,
    "(",
    ws,
    foreign_args(Args),
    ws,
    ")",
    ws,
    ")",
    !.

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
% counts[blob(dyncall...)]++ -- key an assoc table by a runtime
% grammar's byte output (JIT roadmap item 2 follow-on). Must precede
% var(Name): identifier would otherwise eat "blob".
assoc_key_expr(Blob) -->
    blob_call_expr(Blob),
    !.
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

% Whitespace, including newlines and awk-style # comments (a comment
% runs to end of line; its terminating newline still counts as a
% statement separator in action_sep//0). `#` is not a token anywhere
% in the surface, and strings/regex bodies never route through ws//0,
% so comments cannot be consumed inside literals.
ws -->
    [Code],
    { code_type(Code, space) },
    !,
    ws.
ws -->
    "#",
    !,
    comment_rest,
    ws.
ws -->
    [].

comment_rest -->
    [Code],
    { Code =\= 0'\n },
    !,
    comment_rest.
comment_rest -->
    [].

eos([], []).
