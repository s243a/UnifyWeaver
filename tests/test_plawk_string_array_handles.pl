:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Owned string handles for text-valued array elements, and the self-append
% `a[k] = a[k] X` -- docs/design/PLAWK_STRING_STORAGE.md section 4 (S3),
% docs/design/PLAWK_ARRAY_VALUE_MODEL.md plan item 4(b).
%
% A string / strnum element write, a row capture and a row constructor used to
% intern the value, and the atom table never frees: `{ a[$1] = $2 }` kept every
% value ever written, and the self-append (which declined) would have kept every
% prefix -- O(n^2) bytes. A cell now owns a heap handle: an overwrite reuses it,
% an append grows it in place, a delete frees it. The bound is checked under
% `ulimit -v`, as in test_plawk_string_handles.pl. gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 3\nb 5\na 2\nc 1\nb 4\na 7\n").

% N copies of "k abcdefgh"
repeated_input(N, Input) :-
    length(Ls, N),
    maplist(=("k abcdefgh\n"), Ls),
    atomic_list_concat(Ls, Input).

% N records of a 6-digit number and 180 bytes of padding
wide_input(N, Input) :-
    padding(Pad),
    numlist(1, N, Is),
    findall(L, (member(I, Is), format(string(L), "~|~`0t~d~6+ ~w~n", [I, Pad])), Ls),
    atomic_list_concat(Ls, Input).

padding(Pad) :-
    length(Ps, 20), maplist(=("padpadpad"), Ps), atomic_list_concat(Ps, Pad).

long_value(N, Long) :-
    length(Ps, N), maplist(=("abcdefgh"), Ps), atomic_list_concat(Ps, Long).

:- begin_tests(plawk_string_array_handles).

% 8000 appends of 8 bytes to one element: 64 KB live; interning every prefix
% would need about 256 MB. Array-only and beside a scalar (the mixed walker).
test(element_self_append_is_linear, [condition(clang_available)]) :-
    repeated_input(8000, In),
    long_value(8000, Long),
    format(string(Want), "k ~w~n", [Long]),
    run_limited(In, "{ a[$1] = a[$1] $2 } END { for (k in a) print k, a[k] }\n", Want),
    format(string(WantN), "k ~w~n8000~n", [Long]),
    run_limited(In, "{ n++; a[$1] = a[$1] $2 } END { for (k in a) print k, a[k]; print n }\n",
        WantN),
    !.

% 100000 overwrites of one element with a 188-byte value: one live value.
test(element_overwrite_holds_one_value, [condition(clang_available)]) :-
    wide_input(100000, In),
    padding(Pad),
    format(string(Want), "100000 ~w~n", [Pad]),
    run_limited(In, "{ a[$2] = $0 } END { for (k in a) print a[k] }\n", Want),
    run_limited(In, "{ a[$2] = $1 \" \" $2 } END { for (k in a) print a[k] }\n", Want),
    run_limited(In, "{ a[$2] = row($1, $2) } END { for (k in a) print a[k] }\n", Want),
    run_limited(In, "{ a[$2] = $1 } END { for (k in a) print a[k] }\n", "100000\n"),
    !.

test(self_append_forms, [condition(clang_available)]) :-
    run_sorted("{ a[$1] = a[$1] $2 } END { for (k in a) print k, a[k] }\n",
        "a 327\nb 54\nc 1\n"),
    run_sorted("{ a[$1] = a[$1] $2 \",\" } END { for (k in a) print k, a[k] }\n",
        "a 3,2,7,\nb 5,4,\nc 1,\n"),
    run_sorted("{ a[$1] = $2 a[$1] } END { for (k in a) print k, a[k] }\n",
        "a 723\nb 45\nc 1\n"),
    run_sorted("{ a[$1,\"x\"] = a[$1,\"x\"] $2 } END { for (k in a) print a[k] }\n",
        "1\n327\n54\n"),
    run_sorted("{ a[\"all\"] = a[\"all\"] $1 } END { print a[\"all\"] }\n", "abacba\n"),
    !.

% Another element inside the value is not a self-append: it declines.
test(other_element_declines, [condition(clang_available)]) :-
    build_status_is("{ a[$1] = a[$2] $2 } END { for (k in a) print k, a[k] }\n", 3),
    !.

% A delete frees the cell; the key can be written again afterwards.
test(delete_frees_and_rewrites, [condition(clang_available)]) :-
    run_sorted("{ a[$1] = $2; delete a[$1] } END { n = 0; for (k in a) n++; print n }\n",
        "0\n"),
    run_sorted("{ a[$1] = $2 } $1 == \"c\" { delete a[\"a\"] } END { for (k in a) print k, a[k] }\n",
        "a 7\nb 4\nc 1\n"),
    run_sorted("{ a[$1] = a[$1] $2 } $2 == 2 { delete a[$1] } END { for (k in a) print k, a[k] }\n",
        "a 7\nb 54\nc 1\n"),
    !.

% Readers are unchanged: row capture, tac, comparisons on a strnum array.
test(readers_unchanged, [condition(clang_available)]) :-
    run_sorted("{ a[$1] = $0 } END { for (k in a) print k, a[k] }\n",
        "a a 7\nb b 4\nc c 1\n"),
    run_exact("{ a[NR] = $0 } END { for (i = NR; i >= 1; i--) print a[i] }\n",
        "a 7\nb 4\nc 1\na 2\nb 5\na 3\n"),
    run_sorted("{ a[$1] = $2 } END { for (k in a) if (a[k] > 4) print k }\n", "a\n"),
    run_sorted("{ a[$1] = $5 } END { for (k in a) print k, \"[\" a[k] \"]\" }\n",
        "a []\nb []\nc []\n"),
    !.

:- end_tests(plawk_string_array_handles).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_string_array_handles', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_exact(Src, Expected) :-
    input(Input),
    run_output(Input, Src, none, Out),
    check_output(Src, Out, Expected).

run_limited(Input, Src, Expected) :-
    run_output(Input, Src, 16384, Out),
    check_output(Src, Out, Expected).

run_sorted(Src, Expected) :-
    input(Input),
    run_output(Input, Src, none, Out0),
    split_string(Out0, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, Sorted),
    atomic_list_concat(Sorted, '\n', Joined0),
    atom_string(Joined0, Joined),
    string_concat(Joined, "\n", Out),
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
    directory_file_path(Dir, 'sa_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'sa', Prog0),
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
    directory_file_path(Dir, 'sa_st_bin', Bin),
    directory_file_path(Dir, 'sa_st', Prog0),
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
