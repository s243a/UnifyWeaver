:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `delete arr[k]` -- remove an entry from an assoc table (AWK
% `delete arr[k]`). v1 keys on a field (`delete arr[$k]`), matching the counted
% inc `arr[$k]++`; a string-literal / var key is a clean not-yet (compile error).
% The runtime primitive (`@wam_assoc_i64_delete`) does backward-shift deletion
% so later colliding keys stay reachable; a missing key is a no-op. for-in
% iteration order is hash-dependent, so outputs are compared as sorted sets.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_delete).

% --- parsing ----------------------------------------------------------------

test(delete_field_parses) :-
    plawk_parse_string("{ delete a[$1] }\n",
        program([], [rule(always, [delete_assoc(var(a), field(1))])], [])),
    !.

test(delete_guarded_parses) :-
    plawk_parse_string("$1 == \"rm\" { delete counts[$2] }\n",
        program([],
            [rule(field_eq(1, "rm"), [delete_assoc(var(counts), field(2))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% `delete` removes a counted key: `drop` is counted then deleted, so END's
% for-in no longer sees it.
test(delete_removes_key, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ seen[$1]++ }\n$1 == \"drop\" { delete seen[$1] }\nEND { for (k in seen) print k }\n",
    build_run_sorted(Dir, 'rm', Src, "a\nb\ndrop\na\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["a", "b"]),
    !.

% Deleting resets the count; a later insert of the same key counts fresh from 1.
test(delete_resets_count, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ seen[$1]++ }\n$2 == \"x\" { delete seen[$1] }\nEND { for (k in seen) print k, seen[k] }\n",
    build_run_sorted(Dir, 'reset', Src, "a y\na y\na x\na y\nb y\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["a 1", "b 1"]),
    !.

% `delete` may key on a different field than the counter.
test(delete_by_other_field, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ seen[$1]++ }\n$1 == \"rm\" { delete seen[$2] }\nEND { for (k in seen) print k }\n",
    build_run_sorted(Dir, 'other', Src, "a\nb\nrm a\nc\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["b", "c", "rm"]),
    !.

% Deleting an absent key is a no-op (the record's own key stays).
test(delete_missing_is_noop, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ seen[$1]++ }\n$1 == \"q\" { delete seen[$2] }\nEND { for (k in seen) print k, seen[k] }\n",
    build_run_sorted(Dir, 'noop', Src, "a\nb\nq z\na\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["a 2", "b 1", "q 1"]),
    !.

% A group-by with no delete is unaffected.
test(no_delete_unaffected, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ counts[$1]++ }\nEND { for (k in counts) print k, counts[k] }\n",
    build_run_sorted(Dir, 'plain', Src, "a\nb\na\nc\na\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["a 3", "b 1", "c 1"]),
    !.

% A string-literal key is a clean not-yet (v1 keys on a field), so the program
% is rejected with a compile error rather than miscompiled.
test(delete_string_key_rejected, [condition(clang_available)]) :-
    ldir(Dir),
    Src = "{ seen[$1]++ }\n{ delete seen[\"x\"] }\nEND { for (k in seen) print k }\n",
    build_status(Dir, 'strkey', Src, St),
    assertion(St == 3),
    !.

:- end_tests(plawk_delete).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_delete', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin0),
    Bin = Bin0-Prog.

build_status(Dir, Name, Src, Status) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

build_run_sorted(Dir, Name, Src, Input, SortedLines, RunStatus) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)),
    split_string(Out, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, SortedLines).
