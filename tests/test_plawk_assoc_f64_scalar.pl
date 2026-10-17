:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A scalar accumulator in the array-rule (assoc) driver: `n += $2` beside an
% array write. It was an i64 global fed the strict integer parse of the field, so
% "1.5" and "3abc" added 0 (`{ c[$1]++; n += $2; print n }` printed 0, 2, 2, ...).
% A scalar that adds a field is now a double global, accumulated with awk's strtod
% reading and printed the awk way -- nothing before its first add, as gawk prints
% an unset scalar. gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 1.5\nb 2\na 3abc\nc x\nb -0.25\n").

:- begin_tests(plawk_assoc_f64_scalar).

test(field_sum_is_numeric, [condition(clang_available)]) :-
    run("{ c[$1]++; n += $2; print n }\n", "1.5\n3.5\n6.5\n6.5\n6.25\n"),
    run("{ c[$1]++; n += $2; n += 1; print n }\n", "2.5\n5.5\n9.5\n10.5\n11.25\n"),
    run("{ k = $1; c[k]++; n += $2; print k, n }\n",
        "a 1.5\nb 3.5\na 6.5\nc 6.5\nb 6.25\n"),
    !.

test(sum_in_print_arithmetic, [condition(clang_available)]) :-
    run("{ c[$1]++; n += $2; print n * 2, c[$1] }\n",
        "3 1\n7 1\n13 2\n13 1\n12.5 2\n"),
    !.

test(unset_before_first_add_prints_empty, [condition(clang_available)]) :-
    run("$1 == \"b\" { n += $2 } { c[$1]++; print n }\n", "\n2\n2\n2\n1.75\n"),
    !.

% A double is no key id: keying an array by it declines.
test(double_scalar_key_declines) :-
    build_status_is("{ n += $2; c[n]++ } END { for (k in c) print k }\n", 3),
    !.

:- end_tests(plawk_assoc_f64_scalar).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_f64_scalar', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Src, Prog0, Prog) :-
    odir(Dir),
    directory_file_path(Dir, Prog0, P0),
    atom_concat(P0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'fs_st_bin', Bin),
    write_prog(Src, 'fs_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'fs_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'fs', Prog),
    directory_file_path(Dir, 'fs_in.txt', In),
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
