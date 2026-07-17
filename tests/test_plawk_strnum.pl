:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk strnum duality (PLAWK_STRNUM_DUALITY.md).
%
% Step 2: the origin/provenance analysis plawk_scalar_strnum_names/2. A scalar
% is a strnum when it is assigned ONLY from strnum sources (a bare field copy
% `x = $N`) and never from a non-strnum one (a literal, arithmetic, concat, a
% string builtin, another var, ...). This locks down the analysis before the
% codegen activation (slot representation + comparison dispatch) in step 3.
% The runtime primitives (@wam_looks_numeric / @wam_strnum_cmp, step 1) are
% exercised in tests/core/test_wam_llvm_assoc_i64_runtime.pl.

:- use_module(library(plunit)).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

:- begin_tests(plawk_strnum).

strnum_names(Src, Names) :-
    plawk_parse_string(Src, program(_Begin, Rules, _End)),
    plawk_native_codegen:plawk_scalar_strnum_names(Rules, Names).

% a bare field copy is a strnum.
test(field_copy_is_strnum) :-
    strnum_names("{ x = $1 }\n", [x]), !.

% two field copies -> two strnums (sorted).
test(two_field_copies) :-
    strnum_names("{ a = $1; b = $3 }\n", [a, b]), !.

% reading the strnum (print) does not disqualify it.
test(read_does_not_disqualify) :-
    strnum_names("{ x = $1; print x }\n", [x]), !.

% a string-literal assignment is not a strnum.
test(literal_not_strnum) :-
    strnum_names("{ y = \"text\" }\n", []), !.

% an integer-literal assignment is not a strnum.
test(int_literal_not_strnum) :-
    strnum_names("{ x = 5 }\n", []), !.

% a concat assignment (string-valued) is not a strnum.
test(concat_not_strnum) :-
    strnum_names("{ x = $1 $2 }\n", []), !.

% a field copy later overwritten by arithmetic is disqualified (the arithmetic
% write produces a number, destroying the duality).
test(arith_overwrite_disqualifies) :-
    strnum_names("{ x = $1; x = x + 1 }\n", []), !.

% a field copy later overwritten by a literal is disqualified (mixed provenance
% cannot be one static slot kind).
test(literal_overwrite_disqualifies) :-
    strnum_names("{ x = $1; x = \"lit\" }\n", []), !.

% mixed program: the pure field-copy is a strnum, the arithmetic-written name
% and the string name are not.
test(mixed_program) :-
    strnum_names("{ f = $2; c = c + 1; s = \"z\" }\n", [f]), !.

% a strnum plus an unrelated string scalar: only the field copy qualifies.
test(strnum_beside_string) :-
    strnum_names("{ n = $1; s = $2 \"\" }\n", [n]), !.

:- end_tests(plawk_strnum).
