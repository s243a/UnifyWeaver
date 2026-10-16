:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Owned string handles for the remaining per-record writers --
% docs/design/PLAWK_STRING_STORAGE.md, S4: split pieces, sub/gsub results and
% getline reads.
%
% Each of these interned its result, and the atom table never frees, so memory
% grew with the total bytes ever produced (2M records: split 367 MB, the gsub
% stream editor 286 MB, `getline var` 73-144 MB). A split piece is now a handle
% the next split frees; a sub/gsub result is a handle its rule frees once printed
% (or the target variable keeps); a getline line is a handle that reuses the
% target's buffer. Each bound is checked under `ulimit -v`, on records whose
% results are distinct, so interning them would need about 19 MB. gawk 5.1.0
% (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

pad(Pad) :-
    length(Ps, 20), maplist(=("padpadpad"), Ps), atomic_list_concat(Ps, Pad).

% Record I of N: a 6-digit number, a separator and 180 bytes of padding.
wide_line(Sep, I, Line) :-
    pad(Pad),
    format(string(Line), "~|~`0t~d~6+~w~w", [I, Sep, Pad]).

wide_lines(Sep, N, Lines) :-
    numlist(1, N, Is),
    maplist(wide_line(Sep), Is, Lines).

wide_input(Sep, N, Input) :-
    wide_lines(Sep, N, Lines),
    atomic_list_concat(Lines, '\n', Body),
    atom_concat(Body, '\n', A),
    atom_string(A, Input).

:- begin_tests(plawk_string_s4_handles).

% Each split frees the previous pieces. The piece is the whole 188-byte record.
test(split_pieces_are_freed, [condition(clang_available)]) :-
    wide_input(" ", 100000, In),
    wide_line(" ", 100000, Last),
    format(string(Want), "~w~n", [Last]),
    run_limited(In, "{ split($0, p, \"x\") } END { print p[1] }\n", Want),
    run_limited(In, "{ split($0, p, \"x+\") } END { print p[1] }\n", Want),
    !.

% A split target is a text array, so `delete p[k]` frees the piece too.
test(split_piece_delete_frees, [condition(clang_available)]) :-
    wide_input(" ", 100000, In),
    length(Ls, 100000), maplist(=("\n"), Ls), atomic_list_concat(Ls, WantA),
    atom_string(WantA, Want),
    run_limited(In, "{ split($0, p, \"x\"); delete p[1]; print p[1] }\n", Want),
    !.

% The stream editor frees each result after printing it.
test(record_sub_results_are_freed, [condition(clang_available)]) :-
    wide_input(" ", 100000, In),
    wide_lines(" ", 100000, Lines),
    maplist([L, O]>>( sub_string(L, 0, 1, _, "0")
                     -> sub_string(L, 1, _, 0, R), string_concat("Z", R, O)
                     ;  O = L ), Lines, Outs),
    atomic_list_concat(Outs, '\n', Body), atom_concat(Body, '\n', WantA),
    atom_string(WantA, Want),
    run_limited(In, "{ sub(/^0/, \"Z\"); print $0 }\n", Want),
    !.

% A field result stays alive until the record is joined, then is freed.
test(field_sub_results_are_freed, [condition(clang_available)]) :-
    wide_lines("", 100000, Lines),
    maplist([L, I]>>string_concat(L, ",x", I), Lines, InLines),
    maplist([L, O]>>format(string(O), "Z~w x", [L]), Lines, Outs),
    atomic_list_concat(InLines, '\n', InBody), atom_concat(InBody, '\n', InA),
    atom_string(InA, In),
    atomic_list_concat(Outs, '\n', Body), atom_concat(Body, '\n', WantA),
    atom_string(WantA, Want),
    run_limited(In, "BEGIN { FS = \",\" } { sub(/^/, \"Z\", $1); print $0 }\n", Want),
    !.

test(variable_sub_keeps_one_value, [condition(clang_available)]) :-
    wide_input(" ", 100000, In),
    pad(Pad),
    format(string(Want), "1Z0000 ~w~n", [Pad]),
    run_limited(In, "{ s = $0; sub(/0/, \"Z\", s) } END { print s }\n", Want),
    !.

% getline from a file, the main input and a command: the target reuses its buffer.
test(getline_targets_keep_one_value, [condition(clang_available)]) :-
    wide_input(" ", 100000, In),
    side_file(In, Side),
    wide_line(" ", 100000, Last),
    format(string(Want), "~w~n", [Last]),
    format(string(P1), "{ getline x < \"~w\" } END { print x }\n", [Side]),
    run_limited(In, P1, Want),
    format(string(P2), "{ \"cat ~w\" | getline x } END { print x }\n", [Side]),
    run_limited(In, P2, Want),
    format(string(Want3), "50000 ~w~n", [Last]),
    run_limited(In, "{ getline x; n++ } END { print n, x }\n", Want3),
    !.

% A line whose text is "end_of_file" is a line, not the end of the input (the
% reader's EOF marker is that atom; an owned line is told apart by its tag).
test(an_end_of_file_line_is_a_line, [condition(clang_available)]) :-
    In = "a\nend_of_file\nb\nc\n",
    run_exact(In, "{ getline x; print $0, x }\n", "a end_of_file\nb c\n"),
    side_file(In, Side),
    format(string(P), "{ getline x < \"~w\"; print $0, x }\n", [Side]),
    run_exact(In, P, "a a\nend_of_file end_of_file\nb b\nc c\n"),
    !.

% At EOF the target keeps its value (the runtime writes the slot only for a line).
test(eof_keeps_the_target, [condition(clang_available)]) :-
    side_file("s1\n", Side),
    format(string(P), "{ r = getline x < \"~w\"; print r, x }\n", [Side]),
    run_exact("a\nb\nc\n", P, "1 s1\n0 s1\n0 s1\n"),
    !.

% Chained subs on one field: the second reads the first's result, which is
% still alive (freed only after the join).
test(chained_field_subs, [condition(clang_available)]) :-
    run_exact("ab,x\nbb,y\n",
        "BEGIN { FS = \",\" } { sub(/a/, \"b\", $1); gsub(/b/, \"cc\", $1); print $0 }\n",
        "cccc x\ncccc y\n"),
    !.

% A split target's only other write is `delete`: `++` / `+=` would treat a piece
% handle as a number (and mixes key kinds). This used to build, and printed a
% piece's id as a count.
test(split_target_counters_decline) :-
    build_status_is("{ split($1, c, \",\"); c[$1] += 1 } END { for (k in c) print k, c[k] }\n", 3),
    build_status_is("{ split($1, c, \",\"); c[$1]++ } END { for (k in c) print k, c[k] }\n", 3),
    !.

:- end_tests(plawk_string_s4_handles).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_string_s4_handles', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

side_file(Text, Path) :-
    odir(Dir),
    directory_file_path(Dir, 'side.txt', Path),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
        write(S, Text), close(S)).

run_exact(Input, Src, Expected) :-
    run_output(Input, Src, none, Out),
    check_output(Src, Out, Expected).

run_limited(Input, Src, Expected) :-
    run_output(Input, Src, 16384, Out),
    check_output(Src, Out, Expected).

check_output(Src, Out, Expected) :-
    (   Out == Expected
    ->  true
    ;   string_length(Out, GotLen), string_length(Expected, WantLen),
        (   GotLen < 200
        ->  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected])
        ;   format(user_error, "~n~w~n  got ~w bytes, expected ~w~n", [Src, GotLen, WantLen])
        ),
        fail
    ).

% Build Src and run it on Input; with a limit, under `ulimit -v KB` in sh.
run_output(Input, Src, Limit, Out) :-
    odir(Dir),
    directory_file_path(Dir, 's4_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 's4', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    (   Limit == none
    ->  process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)])
    ;   format(atom(Cmd), 'ulimit -v ~w && exec "$0" "$1"', [Limit]),
        process_create(path(sh), ['-c', Cmd, Bin, In],
            [stdout(pipe(PS)), stderr(std), process(Pid)])
    ),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, Exit),
    assertion(Exit == exit(0)).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 's4_st_bin', Bin),
    directory_file_path(Dir, 's4_st', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build status ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).
