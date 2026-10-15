:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% String builders format into a measured heap buffer --
% docs/design/PLAWK_STRING_STORAGE.md section 2.
%
% A string concatenation or sprintf assigned to a scalar (`s = $1 "-" $2`,
% `s = sprintf(...)`) and a row constructor (`a[$3] = row($1, $2)`, the fields
% joined by FS) used to format into a 4096-byte stack buffer allocated in the
% rule body: the value was
% truncated at 4095 bytes, and the stack grew by 4 KiB per record, so a few
% thousand records crashed. The buffer is now sized by a measuring snprintf and
% heap-allocated, interned, then freed. gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% "k1 1 x" .. "kN N x"
numbered_input(N, Input) :-
    numlist(1, N, Is),
    findall(L, (member(I, Is), format(string(L), "k~w ~w x~n", [I, I])), Ls),
    atomic_list_concat(Ls, Input).

% N copies of "k abcdefgh"
repeated_input(N, Input) :-
    length(Ls, N),
    maplist(=("k abcdefgh\n"), Ls),
    atomic_list_concat(Ls, Input).

:- begin_tests(plawk_string_build_heap).

% 20000 records: the old per-record 4 KiB alloca overflowed an 8 MiB stack
% after about 2000.
test(concat_many_records, [condition(clang_available)]) :-
    numbered_input(20000, In),
    run_exact_on(In, "{ s = $1 \"-\" $2 } END { print s }\n", "k20000-20000\n"),
    !.

test(sprintf_many_records, [condition(clang_available)]) :-
    numbered_input(20000, In),
    run_exact_on(In, "{ s = sprintf(\"%s:%d\", $1, $2) } END { print s }\n",
        "k20000:20000\n"),
    !.

% plawk's row constructor -- no gawk equivalent; the fields join with FS.
test(row_cons_many_records, [condition(clang_available)]) :-
    numbered_input(20000, In),
    run_exact_on(In, "{ a[$3] = row($1, $2) } END { for (k in a) print k, a[k] }\n",
        "x k20000 20000\n"),
    !.

% Values past the old 4095-byte cap survive whole.
test(self_concat_past_4k, [condition(clang_available)]) :-
    repeated_input(1000, In),
    length(Ps, 1000), maplist(=("abcdefgh"), Ps),
    atomic_list_concat(Ps, Long),
    format(string(Want), "~w~n", [Long]),
    run_exact_on(In, "{ s = s $2 } END { print s }\n", Want),
    run_exact_on(In, "{ s = sprintf(\"%s%s\", s, $2) } END { print s }\n", Want),
    !.

test(long_field_concat, [condition(clang_available)]) :-
    length(Cs, 6000), maplist(=(0'y), Cs), string_codes(Long, Cs),
    format(string(In), "k ~s~n", [Long]),
    format(string(Want), "<~s>~n", [Long]),
    run_exact_on(In, "{ s = \"<\" $2 \">\" } END { print s }\n", Want),
    !.

% A literal `%` is format text, doubled -- never a directive.
test(literal_percent, [condition(clang_available)]) :-
    numbered_input(3, In),
    run_exact_on(In, "{ s = \"50%s \" $2 } END { print s }\n", "50%s 3\n"),
    !.

:- end_tests(plawk_string_build_heap).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_string_build_heap', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_exact_on(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'sb_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'sb', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
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
    ;   string_length(Out, GotLen), string_length(Expected, WantLen),
        format(user_error, "~n~w~n  got ~w bytes, expected ~w~n", [Src, GotLen, WantLen]),
        fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).
