:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Scalar-variable assoc keys in the PURE-assoc chain -- the no-END group-by idiom
% `{ k = $N; arr[k]++ } END { for (kk in arr) print kk, arr[kk] }`. Unlike the
% mixed chain (which needs an assoc-reading plain-print END and threads scalars
% as SSA slots), the pure-assoc chain (a for-in END) threads the key scalar
% through a module global @plawk_scalar_<name>: a `k = $N` action interns field
% N's slice and stores its atom id in the global, and `arr[k]++` loads that id as
% the key. So the classic word-count works without any END that reads a literal
% key.
%
% v1 scope: the key scalar is set by a field copy (`k = $N`, N >= 1). A computed
% or numeric key in this chain (e.g. `n = NR % 2; c[n]++`) has no field-copy set,
% so it declines cleanly. for-in iteration order is hash-dependent, so outputs
% are compared as sorted sets.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_assoc_varkey_noend).

% --- parsing ----------------------------------------------------------------

test(noend_parses) :-
    plawk_parse_string("{ k = $1; c[k]++ } END { for (kk in c) print kk, c[kk] }\n",
        program([],
            [rule(always, [set(var(k), field(1)), inc_assoc(var(c), var(k))])],
            [end([for_in(var(kk), var(c),
                [print([var(kk), assoc(var(c), var(kk))])])])])),
    !.

% --- runtime ----------------------------------------------------------------

% Word frequency: `{ k = $1; c[k]++ }` counts by a copied field; the for-in END
% dumps the whole table. No literal-key END needed.
test(noend_word_freq, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "{ k = $1; c[k]++ } END { for (kk in c) print kk, c[kk] }\n",
    build_run_sorted(Dir, 'wf', Src, "a\nb\na\nc\na\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["a 3", "b 1", "c 1"]),
    !.

% The key may be copied from a non-first field.
test(noend_key_field2, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "{ k = $2; c[k]++ } END { for (kk in c) print kk, c[kk] }\n",
    build_run_sorted(Dir, 'k2', Src, "r x\nr y\nr x\ns x\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["x 3", "y 1"]),
    !.

% A `==`-guarded rule counts only matching records (parity with a field key).
test(noend_guarded_eq, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "$1 == \"hit\" { k = $1; c[k]++ } END { for (kk in c) print kk, c[kk] }\n",
    build_run_sorted(Dir, 'g', Src, "hit\nx\nhit\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == ["hit 2"]),
    !.

% A missing field copies the empty-string key (awk `arr[""]`, interned to its
% atom id), counted like any other key and resolved back to "" by for-in -- so
% the line is the empty key, OFS, the count. `$2` is empty on single-field records.
test(noend_missing_field_key, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "{ k = $2; c[k]++ } END { for (kk in c) print kk, c[kk] }\n",
    build_run_sorted(Dir, 'mf', Src, "a\nb\na\n", Lines, St),
    assertion(St == 0),
    assertion(Lines == [" 3"]),
    !.

% A numeric/computed key in the pure chain is a clean not-yet: there is no
% field-copy set (`n = NR % 2` is arithmetic), so the program declines with a
% compile error rather than mis-lowering.
test(noend_numeric_key_rejected, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "{ n = NR % 2; c[n]++ } END { for (k in c) print k, c[k] }\n",
    build_status(Dir, 'numk', Src, St),
    assertion(St \== 0),
    !.

:- end_tests(plawk_assoc_varkey_noend).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ndir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_varkey_noend', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin-Prog) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin).

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
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)),
    split_string(Out, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, SortedLines).
