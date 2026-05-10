% =============================================================================
% prolog_term_parser.pl
%
% Portable Prolog term parser. Lives in the cross-target-compilable subset
% (the same constructs the WAM-R / WAM-Haskell / WAM-Scala backends consume),
% so the same source can serve as the canonical spec AND be transpiled into
% any target whose runtime needs to turn a string back into a term -- read/2,
% term_to_atom/2, read_term_from_atom/2,3, command-line arg parsing, etc.
%
% Operator tables are caller-supplied data rather than module-local dynamic
% state. Targets keep their own canonical op storage (e.g. WAM-R holds three
% R lists keyed by fixity) and pass a snapshot in at parse time. This keeps
% the parser pure, compiles cleanly, and matches how the runtime parser in
% templates/targets/r_wam/runtime.R.mustache already works.
%
% API
%   parse_term_from_codes(+Codes, +OpTable, -Term).
%   parse_term_from_atom (+Atom,  +OpTable, -Term).
%   canonical_op_table(-OpTable).
%
% OpTable is a list of op(Name, Prec, Type) facts; Type is one of
% xfx | xfy | yfx | fx | fy | xf | yf.
%
% Variables sharing a source-level name share a Prolog logic variable
% within a single parse; '_' is anonymous (distinct each occurrence).
%
% Compatibility note: this implementation deliberately matches the
% Pratt-style algorithm in WamRuntime$wam_parse_expr. The xf/yf and
% fx/fy distinctions are not enforced under chained postfix / prefix
% application; both treat the operand precedence ceiling as
% `op_prec - 1`. Same simplification documented in the WAM-R handoff.
%
% Compile-target status: this source compiles cleanly to WAM-R
% (every predicate produces a label + wrapper; see
% tests/test_prolog_term_parser_wam_r_compile.pl), and SWI runs it
% to spec. End-to-end equivalence with the inline R parser at
% WamRuntime$parse_term is currently gated on the WAM-R cut
% implementation: the runtime's `!/0` and `CutIte` drop only the
% most-recent CP, which loses cut-barrier semantics whenever a
% multi-clause callee leaves CPs alive before the cut runs (which
% e.g. tokenize_one and take_ident hit). Fix is filed as follow-up
% #3 in docs/handoff/wam_r_session_handoff.md. When that lands the
% inline parser can be retired -- the structural compile test is
% the regression guard.
% =============================================================================

:- module(prolog_term_parser, [
    parse_term_from_codes/3,
    parse_term_from_atom/3,
    canonical_op_table/1
]).

% -----------------------------------------------------------------------------
% Entry points
% -----------------------------------------------------------------------------

parse_term_from_atom(Atom, OpTable, Term) :-
    atom_codes(Atom, Codes),
    parse_term_from_codes(Codes, OpTable, Term).

parse_term_from_codes(Codes, OpTable, Term) :-
    tokenize(Codes, Tokens),
    parse_expr(Tokens, OpTable, 1200, Term, [], _Env, Rest),
    Rest == [].

% -----------------------------------------------------------------------------
% Canonical op table (mirrors WamRuntime$op_infix / op_prefix seeds)
% -----------------------------------------------------------------------------

canonical_op_table([
    op(':-',   1200, xfx),
    op('-->',  1200, xfx),
    op(';',    1100, xfy),
    op('->',   1050, xfy),
    op('*->',  1050, xfy),
    op(',',    1000, xfy),
    op('=',     700, xfx),
    op('\\=',   700, xfx),
    op('==',    700, xfx),
    op('\\==',  700, xfx),
    op('=..',   700, xfx),
    op('is',    700, xfx),
    op('=:=',   700, xfx),
    op('=\\=',  700, xfx),
    op('<',     700, xfx),
    op('>',     700, xfx),
    op('=<',    700, xfx),
    op('>=',    700, xfx),
    op('@<',    700, xfx),
    op('@>',    700, xfx),
    op('@=<',   700, xfx),
    op('@>=',   700, xfx),
    op('+',     500, yfx),
    op('-',     500, yfx),
    op('/\\',   500, yfx),
    op('\\/',   500, yfx),
    op('xor',   500, yfx),
    op('*',     400, yfx),
    op('/',     400, yfx),
    op('//',    400, yfx),
    op('mod',   400, yfx),
    op('rem',   400, yfx),
    op('div',   400, yfx),
    op('<<',    400, yfx),
    op('>>',    400, yfx),
    op('**',    200, xfx),
    op('^',     200, xfy),
    op(':-',   1200, fx),
    op('?-',   1200, fx),
    op('\\+',   900, fy),
    op('-',     200, fy),
    op('+',     200, fy),
    op('\\',    200, fy)
]).

% -----------------------------------------------------------------------------
% Tokenizer
% -----------------------------------------------------------------------------
% codes -> list of:
%   tk_atom(Name)        bare/quoted atoms, single-char solo atoms
%   tk_sym(Name)         symbol-run atoms (`+`, `=:=`, `\\=`, ...)
%   tk_num(Value)        integer or float
%   tk_var(Name)         variable; Name is the source-level identifier
%   tk_lparen / tk_rparen / tk_lbracket / tk_rbracket
%   tk_comma / tk_semicolon / tk_pipe

tokenize(Codes, Tokens) :-
    tokenize_loop(Codes, [], Reverse),
    reverse(Reverse, Tokens).

tokenize_loop([], Acc, Acc).
tokenize_loop([C|Cs], Acc, Out) :-
    (   ws_code(C)
    ->  tokenize_loop(Cs, Acc, Out)
    ;   tokenize_one([C|Cs], Acc, Acc1, Rest),
        tokenize_loop(Rest, Acc1, Out)
    ).

ws_code(32). ws_code(9). ws_code(10). ws_code(13).

% Single-char structural / solo tokens.
tokenize_one([0'(  |R], A, [tk_lparen    |A], R) :- !.
tokenize_one([0')  |R], A, [tk_rparen    |A], R) :- !.
tokenize_one([0'[  |R], A, [tk_lbracket  |A], R) :- !.
tokenize_one([0']  |R], A, [tk_rbracket  |A], R) :- !.
tokenize_one([0',  |R], A, [tk_comma     |A], R) :- !.
tokenize_one([0';  |R], A, [tk_semicolon |A], R) :- !.
tokenize_one([0'|  |R], A, [tk_pipe      |A], R) :- !.
tokenize_one([0'!  |R], A, [tk_atom('!') |A], R) :- !.

% Quoted atom: '...'. Handles \\ and \' escapes; bare \ followed by any
% other char keeps the next char literally (matches the R tokenizer).
tokenize_one([39|R], A, [tk_atom(Atom)|A], Rest) :-
    !,
    take_quoted(R, [], Cs, Rest),
    atom_codes(Atom, Cs).

% Signed number: a `-` followed by a digit, but only when the previous
% emitted token can't combine with `-` as an infix operator (so we don't
% mis-tokenise `X-1` as `X (-1)`). Acc holds the reverse of tokens emitted
% so far.
tokenize_one([0'-, D | R], A, [tk_num(N)|A], Rest) :-
    digit_code(D),
    can_lead_negative(A),
    !,
    take_number_after_sign(D, R, Cs, Rest),
    codes_to_number([0'-|Cs], N).

% Unsigned number.
tokenize_one([D|R], A, [tk_num(N)|A], Rest) :-
    digit_code(D),
    !,
    take_number_after_sign(D, R, Cs, Rest),
    codes_to_number(Cs, N).

% Variable: starts with uppercase or underscore.
tokenize_one([C|R], A, [tk_var(Name)|A], Rest) :-
    var_start_code(C),
    !,
    take_ident([C|R], Cs, Rest),
    atom_codes(Name, Cs).

% Bare atom: lowercase start.
tokenize_one([C|R], A, [tk_atom(Atom)|A], Rest) :-
    lower_code(C),
    !,
    take_ident([C|R], Cs, Rest),
    atom_codes(Atom, Cs).

% Symbol-run atom.
tokenize_one([C|R], A, [tk_sym(Atom)|A], Rest) :-
    sym_code(C),
    !,
    take_syms([C|R], Cs, Rest),
    atom_codes(Atom, Cs).

% Anything else: fail the whole tokenize (unrecognized character).

% --- character classes -------------------------------------------------------

digit_code(C) :- C >= 48, C =< 57.
lower_code(C) :- C >= 97, C =< 122.
upper_code(C) :- C >= 65, C =< 90.
var_start_code(C) :- upper_code(C).
var_start_code(95).                     % `_`
ident_cont(C) :- lower_code(C).
ident_cont(C) :- upper_code(C).
ident_cont(C) :- digit_code(C).
ident_cont(95).                         % `_`

% Same set as the R tokenizer's sym_chars: + - * / \ ^ < > = : @ . ? ~ # $ &.
sym_code(43). sym_code(45). sym_code(42). sym_code(47). sym_code(92).
sym_code(94). sym_code(60). sym_code(62). sym_code(61). sym_code(58).
sym_code(64). sym_code(46). sym_code(63). sym_code(126). sym_code(35).
sym_code(36). sym_code(38).

% --- string-eating helpers ---------------------------------------------------

take_ident([], [], []).
take_ident([C|R], [C|Cs], Rest) :- ident_cont(C), !, take_ident(R, Cs, Rest).
take_ident([C|R], [], [C|R])    :- \+ ident_cont(C).

take_syms([], [], []).
take_syms([C|R], [C|Cs], Rest) :- sym_code(C), !, take_syms(R, Cs, Rest).
take_syms([C|R], [], [C|R])    :- \+ sym_code(C).

% Consume digits, then optional `.digits` for a float. The first digit is
% already consumed by the caller (passed in via Lead); we splice it back.
take_number_after_sign(Lead, R, [Lead|Cs], Rest) :-
    take_digits(R, IntCs, Rest1),
    (   Rest1 = [0'., D | Rest2], digit_code(D)
    ->  take_digits(Rest2, FracCs, Rest),
        append(IntCs, [0'., D | FracCs], Cs)
    ;   Cs = IntCs, Rest = Rest1
    ).

take_digits([], [], []).
take_digits([C|R], [C|Cs], Rest) :- digit_code(C), !, take_digits(R, Cs, Rest).
take_digits([C|R], [], [C|R])    :- \+ digit_code(C).

% Quoted atom body. ' ends the literal; \ acts as a one-char escape so '\''
% and '\\' both round-trip.
take_quoted([39|R], Cs, Cs, R) :- !.
take_quoted([92, C | R], Acc, Out, Rest) :-
    !,
    append(Acc, [C], Acc1),
    take_quoted(R, Acc1, Out, Rest).
take_quoted([C|R], Acc, Out, Rest) :-
    append(Acc, [C], Acc1),
    take_quoted(R, Acc1, Out, Rest).

% codes_to_number: codes -> integer or float. number_codes/2 handles
% both shapes, so the float-vs-int check the original ITE did was
% redundant. Single-clause body avoids the WAM CutIte dependency.
codes_to_number(Cs, N) :-
    number_codes(N, Cs).

% can_lead_negative: a bare `-` followed by a digit is a signed-number lead
% only when the previous emitted token (top of Acc, a reverse stack) is a
% structural separator -- otherwise the `-` is the infix subtraction op.
can_lead_negative([]).
can_lead_negative([T|_]) :- separator_token(T).

separator_token(tk_comma).
separator_token(tk_semicolon).
separator_token(tk_lparen).
separator_token(tk_lbracket).
separator_token(tk_pipe).

% -----------------------------------------------------------------------------
% Parser (Pratt-style)
% -----------------------------------------------------------------------------

% parse_expr(+Tokens, +OpTable, +MaxPrec, -Term, +Env0, -Env, -Rest).
parse_expr(Tokens0, OpTable, MaxPrec, Term, Env0, Env, Rest) :-
    parse_primary(Tokens0, OpTable, MaxPrec, Left, Env0, Env1, Tokens1),
    parse_op_loop(Tokens1, OpTable, MaxPrec, Left, Term, Env1, Env, Rest).

parse_op_loop([], _, _, Left, Left, Env, Env, []).
parse_op_loop([T|Toks], OpTable, MaxPrec, Left, Term, Env0, Env, Rest) :-
    op_loop_step(T, Toks, OpTable, MaxPrec, Left, Term, Env0, Env, Rest).

% op_loop_step: factored out of parse_op_loop to keep the disjunction
% depth shallow -- the WAM compiler's clause_body_analysis pass
% stack-overflows on deeply nested ( A -> B ; C -> D ; E ).
op_loop_step(T, Toks, OpTable, MaxPrec, Left, Term, Env0, Env, Rest) :-
    token_op_name(T, Name),
    op_loop_with_op(Name, Toks, OpTable, MaxPrec, Left, Term, Env0, Env, Rest),
    !.
op_loop_step(T, Toks, _, _, Left, Left, Env, Env, [T|Toks]).

% Tries infix first (matches SWI), falls through to postfix. Fails if
% neither resolution applies, letting op_loop_step backtrack to the
% no-op clause.
op_loop_with_op(Name, Toks, OpTable, MaxPrec, Left, Term, Env0, Env, Rest) :-
    resolve_infix(Name, OpTable, Prec, Type),
    Prec =< MaxPrec,
    !,
    rhs_max_prec(Type, Prec, RhsMax),
    parse_expr(Toks, OpTable, RhsMax, Right, Env0, Env1, Toks1),
    T2 =.. [Name, Left, Right],
    parse_op_loop(Toks1, OpTable, MaxPrec, T2, Term, Env1, Env, Rest).
op_loop_with_op(Name, Toks, OpTable, MaxPrec, Left, Term, Env0, Env, Rest) :-
    resolve_postfix(Name, OpTable, Prec, _Type),
    Prec =< MaxPrec,
    T2 =.. [Name, Left],
    parse_op_loop(Toks, OpTable, MaxPrec, T2, Term, Env0, Env, Rest).

% parse_primary: the leading token of an expression. Number / var / atom /
% list / parenthesised / prefix-op application.
parse_primary([tk_num(V)|R], _, _, V, Env, Env, R) :- !.

parse_primary([tk_var(Name)|R], _, _, V, Env0, Env, R) :- !,
    bind_var(Name, V, Env0, Env).

parse_primary([tk_atom(Name)|R], OpTable, MaxPrec, Term, Env0, Env, Rest) :- !,
    parse_atom_head(Name, R, OpTable, MaxPrec, Term, Env0, Env, Rest).

parse_primary([tk_sym(Name)|R], OpTable, MaxPrec, Term, Env0, Env, Rest) :- !,
    parse_atom_head(Name, R, OpTable, MaxPrec, Term, Env0, Env, Rest).

parse_primary([tk_lparen|R], OpTable, _, Term, Env0, Env, Rest) :- !,
    parse_expr(R, OpTable, 1200, Term, Env0, Env, [tk_rparen|Rest]).

parse_primary([tk_lbracket|R], OpTable, _, Term, Env0, Env, Rest) :- !,
    parse_list_body(R, OpTable, Term, Env0, Env, Rest).

% parse_atom_head: an atom token at primary position can be (1) a functor
% application (atom immediately followed by `(`), (2) a prefix-op application,
% or (3) a stand-alone atom.
parse_atom_head(Name, [tk_lparen|R], OpTable, _, Term, Env0, Env, Rest) :- !,
    parse_args(R, OpTable, Args, Env0, Env, Rest),
    Term =.. [Name|Args].
parse_atom_head(Name, R, OpTable, MaxPrec, Term, Env0, Env, Rest) :-
    R = [Next|_],
    starts_term(Next),
    resolve_prefix(Name, OpTable, Prec),
    Prec =< MaxPrec,
    !,
    OperandMax is Prec - 1,
    parse_expr(R, OpTable, OperandMax, Operand, Env0, Env, Rest),
    Term =.. [Name, Operand].
parse_atom_head(Name, R, _, _, Name, Env, Env, R).

% parse_args: comma-separated arg list inside `(...)`, max_prec 999 so the
% top-level comma operator (1000) doesn't get folded in. Like
% parse_op_loop and list_elem_continue, the post-arg dispatch uses
% clause-level pattern matching rather than nested
% `( A -> B ; C -> D )` -- the WAM compiler doesn't recursively
% recognize Else as another if-then-else and emits the second ->/2
% as a regular Call, which has no runtime implementation. Filed as
% a separate compiler enhancement.
parse_args(Tokens, OpTable, [Arg|Rest], Env0, Env, RestOut) :-
    parse_expr(Tokens, OpTable, 999, Arg, Env0, Env1, Tokens1),
    parse_args_continue(Tokens1, OpTable, Rest, Env1, Env, RestOut).

parse_args_continue([tk_comma|Toks], OpTable, Rest, Env0, Env, RestOut) :-
    !,
    parse_args(Toks, OpTable, Rest, Env0, Env, RestOut).
parse_args_continue([tk_rparen|RestOut], _, [], Env, Env, RestOut).

% parse_list_body: handles `[]`, `[a, b, c]`, `[H|T]`. Element max_prec 999
% (matches parse_args).
parse_list_body([tk_rbracket|R], _, [], Env, Env, R) :- !.
parse_list_body(Tokens, OpTable, List, Env0, Env, Rest) :-
    parse_list_elems(Tokens, OpTable, Elems, Tail, Env0, Env, Rest),
    list_build(Elems, Tail, List).

parse_list_elems(Tokens, OpTable, [E|Rest], Tail, Env0, Env, RestOut) :-
    parse_expr(Tokens, OpTable, 999, E, Env0, Env1, Tokens1),
    list_elem_continue(Tokens1, OpTable, Rest, Tail, Env1, Env, RestOut).

% Three-way clause-level dispatch instead of nested
% `( A -> B ; C -> D ; E )`. The WAM compiler's clause_body_analysis
% pass stack-overflows on the deeply nested disjunction shape (a
% bug separate from the CutIte semantics that motivated the
% other workarounds). Until that's fixed, keep this factored.
list_elem_continue([tk_comma|Toks], OpTable, Rest, Tail, Env0, Env, RestOut) :-
    !,
    parse_list_elems(Toks, OpTable, Rest, Tail, Env0, Env, RestOut).
list_elem_continue([tk_pipe|Toks], OpTable, [], Tail, Env0, Env, RestOut) :-
    !,
    parse_expr(Toks, OpTable, 999, Tail, Env0, Env, Tokens3),
    Tokens3 = [tk_rbracket|RestOut].
list_elem_continue([tk_rbracket|RestOut], _, [], [], Env, Env, RestOut).

list_build([], Tail, Tail).
list_build([E|Es], Tail, [E|Rest]) :- list_build(Es, Tail, Rest).

% --- token / op resolution ---------------------------------------------------

token_op_name(tk_atom(N),    N).
token_op_name(tk_sym(N),     N).
token_op_name(tk_comma,      ',').
token_op_name(tk_semicolon,  ';').

resolve_infix(Name, OpTable, Prec, Type) :-
    member(op(Name, Prec, Type), OpTable),
    is_infix_type(Type),
    !.

resolve_postfix(Name, OpTable, Prec, Type) :-
    member(op(Name, Prec, Type), OpTable),
    is_postfix_type(Type),
    !.

resolve_prefix(Name, OpTable, Prec) :-
    member(op(Name, Prec, Type), OpTable),
    is_prefix_type(Type),
    !.

is_infix_type(xfx).
is_infix_type(xfy).
is_infix_type(yfx).
is_postfix_type(xf).
is_postfix_type(yf).
is_prefix_type(fx).
is_prefix_type(fy).

% rhs_max_prec: xfy keeps the rhs at op_prec (right-associative); xfx/yfx
% drop one (xfx forbids same-prec rhs; yfx folds left through the outer loop
% since each iteration starts at the same MaxPrec ceiling).
rhs_max_prec(xfy, Prec, Prec).
rhs_max_prec(xfx, Prec, Prec1) :- Prec1 is Prec - 1.
rhs_max_prec(yfx, Prec, Prec1) :- Prec1 is Prec - 1.

starts_term(tk_num(_)).
starts_term(tk_var(_)).
starts_term(tk_atom(_)).
starts_term(tk_sym(_)).
starts_term(tk_lparen).
starts_term(tk_lbracket).

% --- variable env ------------------------------------------------------------

% bind_var(+Name, -Var, +Env0, -Env).
%   Looks up Name in Env. If found, unifies Var with the existing logic var.
%   Otherwise adds Name-Var and returns the extended env. The atom '_' is
%   anonymous: every occurrence yields a fresh var.
bind_var('_', _Fresh, Env, Env) :- !.
bind_var(Name, Var, Env, Env) :-
    member(Name-V, Env),
    !,
    Var = V.
bind_var(Name, Var, Env, [Name-Var|Env]).
