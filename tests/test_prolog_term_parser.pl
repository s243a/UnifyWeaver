% =============================================================================
% test_prolog_term_parser.pl
%
% SWI-runnable correctness tests for src/unifyweaver/core/prolog_term_parser.pl.
% The same module will eventually be compiled to one or more WAM targets
% (see test_prolog_term_parser_wam_r_compile.pl for the structural
% compile-to-target test); these tests exercise the algorithm under SWI so
% breakage shows up before any cross-target codegen runs.
% =============================================================================

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/core/prolog_term_parser').

:- begin_tests(prolog_term_parser).

% Helper: parse using the canonical op table.
canon_parse_atom(A, T) :-
    canonical_op_table(Ops),
    parse_term_from_atom(A, Ops, T).

% --- atomic / number / variable primitives ----------------------------------

test(atom_bare) :-
    canon_parse_atom('foo', T),
    T == foo.

test(atom_quoted_with_space) :-
    canon_parse_atom('\'hello world\'', T),
    T == 'hello world'.

test(int_unsigned) :-
    canon_parse_atom('42', T),
    T == 42.

test(int_negative_after_sep) :-
    canon_parse_atom('f(-3)', T),
    T == f(-3).

test(float_basic) :-
    canon_parse_atom('3.14', T),
    T == 3.14.

test(var_uppercase) :-
    canon_parse_atom('X', T),
    var(T).

test(var_underscore_anonymous) :-
    canon_parse_atom('f(_, _)', f(A, B)),
    var(A), var(B),
    A \== B.

test(var_repeated_shares) :-
    canon_parse_atom('f(X, X)', f(A, B)),
    var(A), A == B.

% --- compound terms / lists --------------------------------------------------

test(compound_simple) :-
    canon_parse_atom('foo(a, b)', T),
    T == foo(a, b).

test(compound_nested) :-
    canon_parse_atom('foo(bar(1), baz(X, X))', T),
    T = foo(bar(1), baz(A, B)),
    var(A), A == B.

test(empty_list) :-
    canon_parse_atom('[]', T),
    T == [].

test(list_basic) :-
    canon_parse_atom('[1, 2, 3]', T),
    T == [1, 2, 3].

test(list_pipe_tail) :-
    canon_parse_atom('[a, b | T]', X),
    X = [a, b | Tail],
    var(Tail).

% --- operators --------------------------------------------------------------

test(infix_arith) :-
    canon_parse_atom('1 + 2 * 3', T),
    T == 1 + 2 * 3.

test(infix_yfx_left_assoc) :-
    canon_parse_atom('1 - 2 - 3', T),
    T == (1 - 2) - 3.

test(infix_xfy_right_assoc) :-
    canon_parse_atom('a ; b ; c', T),
    T == (a ; (b ; c)).

test(prefix_minus) :-
    canon_parse_atom('-5', T),
    T == -5.

test(prefix_negation) :-
    canon_parse_atom('\\+ foo', T),
    T == (\+ foo).

test(parenthesised_grouping) :-
    canon_parse_atom('(1 + 2) * 3', T),
    T == (1 + 2) * 3.

% --- postfix --------------------------------------------------------------

postfix_op_table(Ops) :-
    canonical_op_table(Base),
    Ops = [op('!', 100, yf) | Base].

test(postfix_single) :-
    postfix_op_table(Ops),
    parse_term_from_atom('5!', Ops, T),
    T == '!'(5).

test(postfix_chained) :-
    postfix_op_table(Ops),
    parse_term_from_atom('5!!', Ops, T),
    T == '!'('!'(5)).

test(postfix_with_infix) :-
    postfix_op_table(Ops),
    parse_term_from_atom('5! + 3', Ops, T),
    T == '!'(5) + 3.

% --- failure modes ----------------------------------------------------------

test(unterminated_quote_fails, [fail]) :-
    canon_parse_atom('\'oops', _).

test(extra_tokens_fail, [fail]) :-
    canon_parse_atom('a b', _).

test(empty_input_fails, [fail]) :-
    canon_parse_atom('', _).

:- end_tests(prolog_term_parser).
