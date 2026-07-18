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
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
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

% --- read-use gate (step 3): a candidate is deactivated when read in an
% unsupported position, so it stays a plain i64 (no regression) ----------------

% read in arithmetic -> not activated.
test(arith_read_deactivates) :-
    strnum_names("{ x = $1; y = x + 1; print y }\n", []), !.

% compared to a numeric literal -> activated in step 3b (dispatched to
% @wam_strnum_cmp_int), see int_literal_cmp_activates below.

% two field copies compared to each other -> both activated.
test(strnum_vs_strnum_activates) :-
    strnum_names("{ a = $1; b = $2; if (a > b) print \"x\" }\n", [a, b]), !.

% compared to a string literal -> activated (lexical, supported).
test(string_literal_cmp_activates) :-
    strnum_names("{ x = $1; if (x == \"foo\") print \"y\" }\n", [x]), !.

% fixpoint: b is read in arithmetic (deactivated); a then compares against a
% now-deactivated partner, so a is deactivated too.
test(fixpoint_partner_deactivation) :-
    strnum_names("{ a = $1; b = $2; if (a > b) print \"x\"; c = b + 1; print c }\n", []),
    !.

% step 3b: comparison against an integer literal is now a supported read.
test(int_literal_cmp_activates) :-
    strnum_names("{ x = $1; if (x > 3) print \"b\" }\n", [x]), !.

% but arithmetic still deactivates, even alongside an int comparison.
test(int_cmp_but_arith_deactivates) :-
    strnum_names("{ x = $1; if (x > 3) print \"b\"; y = x + 1; print y }\n", []), !.

% --- end-to-end: strnum comparison semantics --------------------------------

% the headline case: two field copies compare numerically when both look
% numeric, lexically otherwise.
test(e2e_numeric_vs_lexical, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nvl', "{ a = $1; b = $2; if (a > b) print \"gt\"; else print \"le\" }\n",
        "10 9\n10 9x\n2 10\n", Out, St),
    assertion(St == 0), assertion(Out == "gt\nle\nle\n"), !.

% printing a strnum prints its field text.
test(e2e_print_text, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ptx', "{ x = $1; print x }\n", "hello\n42\n", Out, St),
    assertion(St == 0), assertion(Out == "hello\n42\n"), !.

% strnum vs string literal is lexical equality.
test(e2e_string_eq, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'seq', "{ x = $1; if (x == \"foo\") print \"eq\"; else print \"ne\" }\n",
        "foo\nbar\n", Out, St),
    assertion(St == 0), assertion(Out == "eq\nne\n"), !.

% a field copy read in arithmetic keeps its numeric behaviour (no regression).
test(e2e_arith_unchanged, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ari', "{ x = $1; y = x + 1; print y }\n", "41\n", Out, St),
    assertion(St == 0), assertion(Out == "42\n"), !.

% a field copy compared to a numeric literal keeps numeric comparison.
test(e2e_numeric_literal_cmp, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nlc', "{ n = $1; if (n == 5) print \"five\" }\n", "5\n6\n", Out, St),
    assertion(St == 0), assertion(Out == "five\n"), !.

% step 3b: strnum vs integer literal -- numeric when the field looks numeric,
% lexical otherwise (a non-numeric field compares as a string against "3").
test(e2e_int_literal_numeric_vs_lexical, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'iln', "{ x = $1; if (x > 3) print \"big\"; else print \"no\" }\n",
        "5\n2\nabc\n", Out, St),
    assertion(St == 0), assertion(Out == "big\nno\nbig\n"), !.

% step 3b: equality vs a number -- "5x" is not the number 5 (lexical), but "05"
% is (numeric).
test(e2e_int_equality, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ieq', "{ n = $1; if (n == 5) print \"y\"; else print \"n\" }\n",
        "5\n05\n5x\n", Out, St),
    assertion(St == 0), assertion(Out == "y\ny\nn\n"), !.

:- end_tests(plawk_strnum).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strnum', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
