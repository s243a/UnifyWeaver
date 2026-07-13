:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-pass persistent cache, phase 3: the cache as the INTER-PASS
% CHANNEL, and durable across runs. A table declared in `BEGIN cache("path")
% { declare NAME }` and shared by the passes is loaded from its store before
% pass 1 and committed after the last pass -- so the same table is both the
% in-memory channel between passes (already the multi-pass model) AND durable
% between separate runs of the binary. Re-running the binary loads the prior
% committed state and accumulates onto it. Works for the file backend
% (default) and, when liblmdb is present, `backend "lmdb"`; and for both the
% no-END normalise shape and the END for-in report. See
% PLAWK_MULTIPASS_CACHE.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

clang_lmdb_available :-
    catch(( cdir(Dir), directory_file_path(Dir, 'probe.c', C),
            directory_file_path(Dir, 'probe_bin', Bin),
            setup_call_cleanup(open(C, write, S),
                write(S, "#include <lmdb.h>\nint main(void){MDB_env*e;return mdb_env_create(&e);}\n"),
                close(S)),
            format(atom(Cmd), 'clang -w ~w -o ~w -llmdb 2>/dev/null', [C, Bin]),
            process_create(path(sh), ['-c', Cmd], [process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_multipass_cache).

% File backend, no-END normalise shape: `total[$1] += $2` folded in pass 1
% into a cache-backed table, read back per record in pass 2. The store
% persists, so each run adds another copy of the input to the totals.
% Over a=10, b=5: run 1 -> a 10 / b 5; run 2 -> a 20 / b 10.
test(file_backend_persists, [condition(clang_available)]) :-
    cdir(Dir),
    fresh(Dir, 'mpc.db'),
    directory_file_path(Dir, 'mpc.db', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare total }\n\c
         pass { total[$1] += $2 }\n\c
         pass { print $1, total[$1] }\n", [Store]),
    build(Dir, 'mpc', Src, Bin, In, "a 10\nb 5\n"),
    run_sorted(Bin, In, S1), assertion(S1 == ["a 10", "b 5"]),
    run_sorted(Bin, In, S2), assertion(S2 == ["a 20", "b 10"]),
    run_sorted(Bin, In, S3), assertion(S3 == ["a 30", "b 15"]),
    !.

% File backend, END for-in report: two passes each count every record, so a
% single run doubles the counts; the store persists across runs. Over a a b:
% run 1 -> a 4 / b 2; run 2 -> a 8 / b 4.
test(file_backend_end_forin, [condition(clang_available)]) :-
    cdir(Dir),
    fresh(Dir, 'mpe.db'),
    directory_file_path(Dir, 'mpe.db', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare c }\n\c
         pass { c[$1]++ }\n\c
         pass { c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n", [Store]),
    build(Dir, 'mpe', Src, Bin, In, "a\na\nb\n"),
    run_sorted(Bin, In, S1), assertion(S1 == ["a 4", "b 2"]),
    run_sorted(Bin, In, S2), assertion(S2 == ["a 8", "b 4"]),
    !.

% LMDB backend, same durability across runs (skipped without liblmdb).
test(lmdb_backend_persists,
        [condition((clang_available, clang_lmdb_available))]) :-
    cdir(Dir),
    fresh_lmdb(Dir, 'mpl.lmdb'),
    directory_file_path(Dir, 'mpl.lmdb', Store),
    format(atom(Src),
        "BEGIN cache(\"~w\" backend \"lmdb\") { declare c }\n\c
         pass { c[$1]++ }\n\c
         pass { print $1, c[$1] }\n", [Store]),
    build(Dir, 'mpl', Src, Bin, In, "a\na\nb\n"),
    run_sorted(Bin, In, S1), assertion(S1 == ["a 2", "a 2", "b 1"]),
    run_sorted(Bin, In, S2), assertion(S2 == ["a 4", "a 4", "b 2"]),
    !.

:- end_tests(plawk_multipass_cache).

% --- helpers ---------------------------------------------------------------

cdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_multipass_cache', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

fresh(Dir, Name) :-
    directory_file_path(Dir, Name, F),
    ( exists_file(F) -> delete_file(F) ; true ).

% LMDB with MDB_NOSUBDIR leaves a "<name>-lock" sibling too.
fresh_lmdb(Dir, Name) :-
    fresh(Dir, Name),
    atom_concat(Name, '-lock', Lock),
    fresh(Dir, Lock).

build(Dir, Name, Src, Bin, In, Input) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)).

run_sorted(Bin, In, Sorted) :-
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
