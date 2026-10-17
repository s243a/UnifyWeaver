:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A scalar against a FLOAT literal, and a field-copied (strnum) scalar in float
% arithmetic -- docs/design/PLAWK_SCALAR_VALUE_MODEL.md.
%
% `x = $2; if (x > 5.5)` declined, and so did a counter or double scalar against a
% float anywhere but a bare rule pattern (`if (t > 5.5)`, `while (t > 5.5)`, END).
% `x = $2; print x + 0.5` printed 0.5 for "9x" (awk's strtod reads 9): the float
% literal took x off the strnum path onto a strict integer parse. And a bare field
% copy into a NUMERIC slot (`v = $2; v = v * 2`) read "10.5" as 0; that shape now
% declines. gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% "x" and "9x" do not look numeric.
input("a 3\nb 10\nc x\nd 9x\ne 5.5\nf 5.6\na 7\nh -1.5\n").

:- begin_tests(plawk_strnum_scalar_float).

% x reads as its strtod value: "9x" is 9, "x" is 0.
test(strnum_float_arithmetic, [condition(clang_available)]) :-
    run("{ x = $2; print x + 0.5 }\n", "3.5\n10.5\n0.5\n9.5\n6\n6.1\n7.5\n-1\n"),
    run("{ x = $2; y = x * 2.5; print y }\n",
        "7.5\n25\n0\n22.5\n13.75\n14\n17.5\n-3.75\n"),
    !.

% "x" > 5.5 and "9x" > 5.5 compare as strings (true); the numeric texts numerically.
test(strnum_float_conditions, [condition(clang_available)]) :-
    run("{ x = $2; if (x > 5.5) print $1 }\n", "b\nc\nd\nf\na\n"),
    run("{ x = $2; if (x == 5.5) print $1 }\n", "e\n"),
    run("{ x = $2; if (x < -1.25) print $1 }\n", "h\n"),
    run("{ x = $2; if (x >= 5.5) n++ } END { print n+0 }\n", "6\n"),
    run("{ x = $2 } x > 5.5 { print $1 }\n", "b\nc\nd\nf\na\n"),
    !.

test(end_float_conditions, [condition(clang_available)]) :-
    run("{ x = $2 } END { if (x > 5.5) print \"big\"; else print \"small\" }\n", "small\n"),
    run("{ x = $2 } END { if (x == -1.5) print \"neg\" }\n", "neg\n"),
    run("{ t += $2 } END { if (t > 5.5) print \"y\"; else print \"n\" }\n", "y\n"),
    !.

test(numeric_scalar_float_conditions, [condition(clang_available)]) :-
    run("{ n++; if (n > 2.5 && n < 5.5) print $1 }\n", "c\nd\ne\n"),
    run("{ t += $2; while (t > 5.5) t = t - 10; print t }\n",
        "3\n3\n3\n2\n-2.5\n3.1\n0.1\n-1.4\n"),
    !.

% x written both a field and a number holds whichever came last; one static
% numeric slot kind cannot, so these decline (they printed 0 for "10.5").
test(field_copy_into_numeric_slot_declines) :-
    build_status_is("{ v = $2; v = v * 2; print v }\n", 3),
    build_status_is("{ x = 0; x = $2; print x }\n", 3),
    build_status_is("{ x = $2; if (x > 5) x = 0; print x }\n", 3),
    !.

:- end_tests(plawk_strnum_scalar_float).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_strnum_scalar_float', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

write_prog(Src, Prog0, Prog) :-
    odir(Dir),
    directory_file_path(Dir, Prog0, P0),
    atom_concat(P0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'sf_st_bin', Bin),
    write_prog(Src, 'sf_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'sf_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'sf', Prog),
    directory_file_path(Dir, 'sf_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, Exit),
    assertion(Exit == exit(0)),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).
