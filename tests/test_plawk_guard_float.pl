:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk reader-guard extensions -- FLOAT literals (follow-on to the integer
% reader guards). A row-reader guard `if (COL CMP LIT)` now accepts a decimal
% float literal on the right, e.g. `if (r["amt"] > 3.5)`: the column is
% extracted as a double (@wam_atom_field_f64_value) and compared with fcmp
% against the exact decimal ratio, so fractional thresholds and fractional
% column values compare correctly. A bare-integer RHS is unchanged (i64 icmp).
% Signed floats (`-1.5`) are supported. Applies to all three readers -- named
% `r["col"]`, positional `r[N]`, and awk-native `$N`.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_guard_float).

% A float RHS parses to a float_const(Mantissa, Denominator); a bare integer is
% unchanged.
test(float_guard_parses) :-
    plawk_parse_string(
        "pass rows of t as r { if (r[2] > 3.5) print r[1] }\n",
        program_passes([],
            [pass_rows(var(r), var(t),
                [if(rpos_cmp(r, 2, gt, float_const(35, 10)),
                    [print([assoc(var(r), int(1))])], [])])],
            [])),
    plawk_parse_string(
        "pass rows of t as r { if (r[2] > 3) print r[1] }\n",
        program_passes([],
            [pass_rows(var(r), var(t),
                [if(rpos_cmp(r, 2, gt, 3),
                    [print([assoc(var(r), int(1))])], [])])],
            [])),
    !.

% A negative float RHS parses to a negative mantissa.
test(negative_float_parses) :-
    plawk_parse_string(
        "pass rows of t { if ($2 <= -1.5) print $1 }\n",
        program_passes([],
            [pass_rows_anon(var(t),
                [if(rfield_cmp(2, le, float_const(-15, 10)),
                    [print([field(1)])], [])])],
            [])),
    !.

% Positional float guard: r[2] > 3.5 over integer-looking columns.
test(rows_float_guard, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t as r { if (r[2] > 3.5) print r[1] }\n",
    run_sorted(Dir, 'rf', Src, "a 4\nb 2\nc 10\n", S),
    assertion(S == ["a", "c"]),
    !.

% Named float guard over FRACTIONAL column values: amt >= 2.5.
test(records_float_guard_fractional, [condition(clang_available)]) :-
    gdir(Dir),
    directory_file_path(Dir, 'ff.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare t(k str, amt str) }\n\c
         pass { t[$1] = row($1, $2) }\n\c
         pass records of t as r { if (r[\"amt\"] >= 2.5) print r[\"k\"] }\n", [Store]),
    run_sorted(Dir, 'rff', Src, "a 2.5\nb 1.2\nc 9.9\n", S),
    assertion(S == ["a", "c"]),
    !.

% Awk-native `$N` with a negative float threshold.
test(anon_negative_float_guard, [condition(clang_available)]) :-
    gdir(Dir),
    Src = "pass { t[$1] = $0 }\npass rows of t { if ($2 > -1.5) print $1 }\n",
    run_sorted(Dir, 'anf', Src, "x 3\ny -2\nz 0\n", S),
    assertion(S == ["x", "z"]),
    !.

% All six operators against a float threshold (via the anon reader), over
% 4 / 2 / 10 with threshold 4.0.
test(float_op_lt, [condition(clang_available)]) :- fop("$2 < 4.0", ["b"]).
test(float_op_le, [condition(clang_available)]) :- fop("$2 <= 4.0", ["a", "b"]).
test(float_op_gt, [condition(clang_available)]) :- fop("$2 > 4.0", ["c"]).
test(float_op_ge, [condition(clang_available)]) :- fop("$2 >= 4.0", ["a", "c"]).
test(float_op_eq, [condition(clang_available)]) :- fop("$2 == 4.0", ["a"]).
test(float_op_ne, [condition(clang_available)]) :- fop("$2 != 4.0", ["b", "c"]).

:- end_tests(plawk_guard_float).

% --- helpers ---------------------------------------------------------------

fop(Cond, Expected) :-
    gdir(Dir),
    format(atom(Src),
        "pass { t[$1] = $0 }\npass rows of t { if (~w) print $1 }\n", [Cond]),
    run_sorted(Dir, 'fop', Src, "a 4\nb 2\nc 10\n", S),
    assertion(S == Expected).

gdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_guard_float', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Dir, Name, Src, Input, Sorted) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
