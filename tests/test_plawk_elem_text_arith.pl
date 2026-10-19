:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% An element of a STRING-kind table (`a[k] = $2 ""`) as an operand of arithmetic
% in a rule body: `t += a[$1]`, `x = a[$1] * 2`. Arithmetic reads it as awk's
% strtod of its text ("9x" is 9, "x" is 0), exactly as it reads a strnum
% element; only a comparison tells the two kinds apart (`a[$1] > 4` compares
% "10" and "4" as strings). These declined; `print a[$1] + 1` already worked.
%
% Also: `length(n)` of a numeric scalar in END emitted invalid IR (clang failed);
% it declines now, like the rule-body form.
%
% gawk 5.1.0 (LC_ALL=C) gives every expected output below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% "x" and "9x" do not look numeric.
input(fields, "a 3\nb 10\nc x\nd 9x\ne 5.5\na 7\n").
input(empty, "").

:- begin_tests(plawk_elem_text_arith).

test(accumulate_string_element, [condition(clang_available)]) :-
    run(fields, "{ a[$1] = $2 \"\"; t += a[$1] } END { print t }\n", "34.5\n"),
    run(fields, "{ a[$1] = $2 \"\"; t += a[$1] * 2 } END { print t }\n", "69\n"),
    run(fields, "{ a[$1] = $2 \"\"; t += a[$1]; print t, a[$1] }\n",
        "3 3\n13 10\n13 x\n22 9x\n27.5 5.5\n34.5 7\n"),
    run(fields, "{ a[$1] = $1 \"-\" $2; t += a[$1] } END { print t+0 }\n", "0\n"),
    run(fields, "{ a[$1] = $2 \"\"; t += a[\"zz\"] } END { print t }\n", "0\n"),
    run(empty, "{ a[$1] = $2 \"\"; t += a[$1] } END { print t }\n", "\n"),
    !.

test(assign_from_string_element, [condition(clang_available)]) :-
    run(fields, "{ a[$1] = $2 \"\"; x = a[$1] * 2; print x }\n",
        "6\n20\n0\n18\n11\n14\n"),
    run(fields, "{ a[$1] = $2 \"\"; x = a[$1] + 0.5; print x }\n",
        "3.5\n10.5\n0.5\n9.5\n6\n7.5\n"),
    run(fields, "{ a[$1] = $2 \"\"; y = a[$1] * a[$1]; print y }\n",
        "9\n100\n0\n81\n30.25\n49\n"),
    run(fields, "{ a[$1] = $2 \"\"; b[$1] = $2; t += a[$1] + b[$1] } END { print t }\n",
        "69\n"),
    !.

% the comparison reads the element as a string ("10" < "4"), the sum as a number
test(compare_as_string_add_as_number, [condition(clang_available)]) :-
    run(fields, "{ a[$1] = $2 \"\"; if (a[$1] > 4) t += a[$1] } END { print t }\n",
        "21.5\n"),
    !.

test(end_length_of_number_declines) :-
    build_status("{ t += $2 } END { print length(t) }\n", 3),
    build_status("{ n++ } END { print length(n) }\n", 3),
    !.

:- end_tests(plawk_elem_text_arith).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_elem_text_arith', Dir),
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

build_status(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'ea_st_bin', Bin),
    write_prog(Src, 'ea_st', Prog),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

run(InputName, Src, Expected) :-
    odir(Dir),
    input(InputName, Input),
    directory_file_path(Dir, 'ea_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    write_prog(Src, 'ea', Prog),
    directory_file_path(Dir, 'ea_in.txt', In),
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
