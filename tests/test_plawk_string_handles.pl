:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Owned string handles for string and strnum scalars --
% docs/design/PLAWK_STRING_STORAGE.md section 3.
%
% A string or strnum slot used to hold an interned atom id, and the atom table
% never frees: `{ last = $0 }` kept every record, `{ s = s $2 }` every prefix of
% s (quadratic). A slot written at run time now owns a heap handle that
% @wam_atom_to_string resolves, so the readers are unchanged; an overwrite reuses
% it, an append grows it in place, a copy duplicates it. The bound is checked
% under `ulimit -v`: past the limit the old binaries printed short or empty
% output (an intern that cannot allocate yields atom 0). gawk 5.1.0 (LC_ALL=C).

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
    length(Ps, 20), maplist(=("padpadpad"), Ps), atomic_list_concat(Ps, Pad),
    numlist(1, N, Is),
    findall(L, (member(I, Is), format(string(L), "~|~`0t~d~6+ ~w~n", [I, Pad])), Ls),
    atomic_list_concat(Ls, Input).

:- begin_tests(plawk_string_handles).

% 8000 appends of 8 bytes: 64 KB live. Interning every prefix needed about
% 256 MB; the handle grows in place.
test(self_append_is_linear, [condition(clang_available)]) :-
    repeated_input(8000, In),
    length(Ps, 8000), maplist(=("abcdefgh"), Ps), atomic_list_concat(Ps, Long),
    format(string(Want), "~w~n", [Long]),
    run_exact_limited(In, "{ s = s $2 } END { print s }\n", Want),
    !.

% 100000 overwrites of a 188-byte record: one live value. Interning kept all
% 19 MB of them.
test(overwrite_holds_one_value, [condition(clang_available)]) :-
    wide_input(100000, In),
    length(Ps, 20), maplist(=("padpadpad"), Ps), atomic_list_concat(Ps, Pad),
    format(string(Want), "100000 ~w~n", [Pad]),
    run_exact_limited(In, "{ last = $0 } END { print last }\n", Want),
    run_exact_limited(In, "{ s = $1 \" \" $2 } END { print s }\n", Want),
    !.

test(append_prepend_and_sprintf_forms, [condition(clang_available)]) :-
    run_exact("{ s = s $1 } END { print s }\n", "abacba\n"),
    run_exact("{ s = $1 s } END { print s }\n", "abcaba\n"),
    run_exact("{ s = s $1 \",\" } END { print s }\n", "a,b,a,c,b,a,\n"),
    run_exact("{ s = sprintf(\"%s%s\", s, $1) } END { print s }\n", "abacba\n"),
    run_exact("{ s = s $2 \",\"; s = s \";\" } END { print s }\n",
        "3,;5,;2,;1,;4,;7,;\n"),
    run_exact("BEGIN { s = \"x\" } { s = s $1 } END { print s }\n", "xabacba\n"),
    !.

% A copy owns its own bytes: rewriting the source (its buffer reused in place)
% leaves the copy alone. Only strnum copies compile; a string-to-string copy
% (`t = s` of a concatenation) still declines.
test(copies_are_independent, [condition(clang_available)]) :-
    run_exact("{ t = s; s = $1 } END { print s, t }\n", "a b\n"),
    run_exact("{ x = $1; y = x; x = $2; print x, y }\n",
        "3 a\n5 b\n2 a\n1 c\n4 b\n7 a\n"),
    run_exact("{ s = $1; s = s \"!\"; print s }\n", "a!\nb!\na!\nc!\nb!\na!\n"),
    !.

% String comparisons go by text, never by id: equal text in two handles, a
% handle against a literal, and ordering (`s < t` used to order by intern id).
test(comparisons_by_text, [condition(clang_available)]) :-
    run_exact("{ s = $1 \"\"; t = $1 \"\"; if (s == t) n++ } END { print n }\n", "6\n"),
    run_exact("{ s = $1 \"\"; if (s == \"a\") n++ } END { print n }\n", "3\n"),
    run_exact("{ s = $1 \"\"; t = \"b\"; if (s < t) print s }\n", "a\na\na\n"),
    run_exact("{ x = $2; if (x > 4) print $1 }\n", "b\na\n"),
    !.

% A string variable as an array key is interned (table keys are atom ids).
test(handle_as_key, [condition(clang_available)]) :-
    run_sorted("{ k = $1 \"\"; c[k]++ } END { for (q in c) print q, c[q] }\n",
        "a 3\nb 2\nc 1\n"),
    run_exact("{ k = $1 \"\"; c[k]++; if (c[k] > 2) print k, \"thrice\" }\n",
        "a thrice\n"),
    run_exact("{ k = $1 \"\"; print k, c[k]; c[k]++ }\n", "a \nb \na 1\nc \nb 1\na 2\n"),
    !.

test(gsub_on_a_handle, [condition(clang_available)]) :-
    run_exact("{ s = $1 \"\"; gsub(/a/, \"X\", s); print s }\n", "X\nb\nX\nc\nb\nX\n"),
    !.

% A string variable against a NUMERIC one has no lowering yet: it declines
% instead of comparing an id with a number.
test(string_vs_number_declines, [condition(clang_available)]) :-
    build_status_is("{ t = $2 \"\"; u = $2; if (t > u) print \"gt\" }\n", 3),
    !.

:- end_tests(plawk_string_handles).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_string_handles', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_exact(Src, Expected) :-
    input(Input),
    run_output(Input, Src, none, Out),
    check_output(Src, Out, Expected).

run_exact_limited(Input, Src, Expected) :-
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
    directory_file_path(Dir, 'sh_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'sh', Prog0),
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
    directory_file_path(Dir, 'sh_st_bin', Bin),
    directory_file_path(Dir, 'sh_st', Prog0),
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
