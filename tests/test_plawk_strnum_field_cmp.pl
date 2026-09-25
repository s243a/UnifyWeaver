:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `$N OP int` with awk (POSIX) strnum semantics -- phase B of the numeric-fields fix.
%
% ---------------------------------------------------------------------------
% A field is input, so it is a STRNUM: compared against a number it is compared
% NUMERICALLY when its text looks numeric (blanks, sign, decimal digits/point,
% exponent -- the whole text), and otherwise as a STRING against the number's text.
% A missing or empty field is the string "".
%
% plawk parsed the field as a strict integer and made the comparison FALSE on any
% parse failure, so both halves of the rule were wrong:
%
%     "30.25" > 10    numeric in awk: true     plawk: false
%     "3abc"  > 10    string  in awk: true     plawk: false  ("3abc" > "10")
%     ""      < 1     string  in awk: true     plawk: false  (a missing field)
%
% Now every `$N OP int` surface -- the rule pattern, the multi-rule guard, `$N OP
% counter`, and `if`/ternary conditions -- goes through one runtime,
% @wam_atom_field_strnum_cmp_int: the strict integer parse is the fast path, and
% anything else is decided by the existing POSIX @wam_strnum_cmp_int.
% @wam_looks_numeric is now DECIMAL only, like the value read (@wam_awk_strtod):
% "0x1A", "inf", "nan" are strings, as in gawk (the signed "+inf" gawk accepts as a
% number is a deliberate, negligible divergence).
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% $2 per line: 30.25, 5, 3abc, 12 (blank-surrounded), 0x1A, inf, missing, 1e3, -
input("a 30.25\nb 5\nc 3abc\nd  12 \ne 0x1A\nf inf\ng\nh 1e3\ni -\n").

:- begin_tests(plawk_strnum_field_cmp).

test(greater_than, [condition(clang_available)]) :-
    run("$2 > 10 { print $1 }\n", "a\nc\nd\nf\nh\n"),
    !.

test(less_than_includes_string_compares, [condition(clang_available)]) :-
    % "3abc" < "5", "0x1A" < "5", "" < "5", "-" < "5" -- all string comparisons
    run("$2 < 5 { print $1 }\n", "c\ne\ng\ni\n"),
    !.

test(equality, [condition(clang_available)]) :-
    run("$2 == 5 { print $1 }\n", "b\n"),
    !,
    run("$2 != 5 { print $1 }\n", "a\nc\nd\ne\nf\ng\nh\ni\n"),
    !,
    run("$2 >= 12 { print $1 }\n", "a\nc\nd\nf\nh\n"),
    !.

% A missing field is "": a string comparison.
test(missing_field_is_the_empty_string, [condition(clang_available)]) :-
    run("$3 < 1 { n++ } END { print n }\n", "9\n"),
    !,
    run("$3 == 0 { n++ } END { print n + 0 }\n", "0\n"),
    !.

test(field_vs_counter, [condition(clang_available)]) :-
    run("{ n++ } $2 > n { print $1 }\n", "a\nb\nc\nd\nf\nh\n"),
    !.

test(combinators, [condition(clang_available)]) :-
    run("$2 > 10 && $1 != \"a\" { print $1 }\n", "c\nd\nf\nh\n"),
    !.

test(if_and_ternary_agree_with_the_pattern, [condition(clang_available)]) :-
    run("{ if ($2 > 10) print $1 }\n", "a\nc\nd\nf\nh\n"),
    !,
    run("{ print ($2 > 10) ? \"big\" : \"small\" }\n",
        "big\nsmall\nbig\nbig\nsmall\nbig\nsmall\nbig\nsmall\n"),
    !,
    run("{ print (10 < $2) ? \"big\" : \"small\" }\n",
        "big\nsmall\nbig\nbig\nsmall\nbig\nsmall\nbig\nsmall\n"),
    !,
    run("{ x = $2 > 10 ? 1 : 0; print x }\n", "1\n0\n1\n1\n0\n1\n0\n1\n0\n"),
    !.

% Plain integer text keeps the exact fast path (no string/float round trip). gawk
% compares through doubles, so 2^53+1 and 2^53 are EQUAL to it and it prints
% nothing here; plawk's integers are exact i64 -- a deliberate, pre-existing
% difference (also pinned in test_plawk_field_numeric_value.pl).
test(integer_text_fast_path_unchanged, [condition(clang_available)]) :-
    run_with("a 9007199254740993\nb 9007199254740992\n",
        "$2 > 9007199254740992 { print $1 }\n", "a\n"),
    !.

test(ir_uses_the_strnum_runtime) :-
    ir_of("$2 > 10 { print }\n", IR1),
    sub_atom(IR1, _, _, _, '@wam_atom_field_strnum_cmp_int(%Value %line, i64 2, i8 32, i64 10, i32 4)'),
    \+ sub_atom(IR1, _, _, _, '@wam_atom_field_i64_cmp_value('),
    ir_of("{ print ($2 > 10) ? 1 : 0 }\n", IR2),
    sub_atom(IR2, _, _, _, '@wam_atom_field_strnum_cmp_int('),
    !.

:- end_tests(plawk_strnum_field_cmp).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strnum_field_cmp', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'sfc_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'sfc', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(BO)), stderr(null), process(BPid)]),
    read_string(BO, _, _), close(BO),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

ir_of(Src, IR) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR).
