:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Scalar-variable keys in associative arrays: `arr[x]++` / `delete arr[x]` where
% `x` is a scalar variable (the group-by-a-copied-field idiom,
% `{ k = $1; freq[k]++ }`). Covered in the mixed chain (a program whose END is a
% plain print, so scalar slots are threaded to the assoc-key site): the key
% scalar must be a string/strnum slot -- a field-assigned scalar (`k = $1`),
% whose slot value already IS the interned atom id `arr[$1]` would produce, so
% `arr[k]` uses it directly (no field-slice / re-intern). Reading the scalar as
% an assoc key (inc or delete) is a supported strnum read, so `k` stays a strnum
% slot rather than deactivating to a numeric counter.
%
% INC (`arr[k]++`), DELETE (`delete arr[k]`), and VALUE READ (`print arr[k]`)
% all work. The read fetches the i64 count for the resolved key id via
% @wam_assoc_i64_get and prints it with %ld (an absent key is 0, awk's
% numeric-context default); it composes in a comma list and in concat
% (`print "n=" arr[k]`). Because the mixed chain still needs an assoc-reading
% plain-print END, the read tests carry an `END { print arr["lit"] }`.
%
% A numeric/counter scalar key (`{ n++; arr[n]++ }` / `delete arr[n]` /
% `print arr[n]`) is a clean not-yet (no i64->string key path) -- it declines
% with a compile error rather than mis-lowering. Reading via `printf` and reading
% in a no-END program are separate follow-ons (both decline cleanly for now).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_assoc_varkey).

% --- parsing ----------------------------------------------------------------

test(varkey_inc_parses) :-
    plawk_parse_string("{ k = $1; a[k]++ }\n",
        program([], [rule(always,
            [set(var(k), field(1)), inc_assoc(var(a), var(k))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% `{ k = $1; freq[k]++ }` groups by a copied field; END looks up known keys.
test(varkey_inc_counts, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vk1', "{ k = $1; freq[k]++ } END { print freq[\"a\"] }\n",
        "a\nb\na\n", Out),
    assertion(Out == "2\n"), !.

% Keys on a non-first field.
test(varkey_inc_other_field, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vk2', "{ k = $2; c[k]++ } END { print c[\"x\"], c[\"y\"] }\n",
        "r x\nr y\nr x\n", Out),
    assertion(Out == "2 1\n"), !.

% A word-frequency count: several distinct keys, multiple lookups.
test(varkey_word_freq, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vk3', "{ w = $1; freq[w]++ } END { print freq[\"the\"], freq[\"cat\"] }\n",
        "the\ncat\nthe\nthe\n", Out),
    assertion(Out == "3 1\n"), !.

% The key scalar's value updates per record (a fresh copy each line).
test(varkey_reassigned_each_record, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vk4', "{ k = $1; seen[k]++ } END { print seen[\"p\"], seen[\"q\"] }\n",
        "p\nq\np\nq\np\n", Out),
    assertion(Out == "3 2\n"), !.

% --- delete by a scalar-variable key ----------------------------------------

test(varkey_delete_parses) :-
    plawk_parse_string("{ k = $1; delete a[k] }\n",
        program([], [rule(always,
            [set(var(k), field(1)), delete_assoc(var(a), var(k))])], [])),
    !.

% `delete arr[k]` resets the count: the key is counted then deleted, so a later
% insert of the same key counts fresh from 1.
test(varkey_delete_resets_count, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkd1',
        "{ k = $1; c[k]++ }\n$2 == \"x\" { k = $1; delete c[k] }\nEND { print c[\"a\"] }\n",
        "a p\na p\na x\na p\n", Out),
    assertion(Out == "1\n"), !.

% The delete key may come from a different scalar than the counter (here `$2`).
test(varkey_delete_by_other_scalar, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkd2',
        "{ k = $1; c[k]++ }\n$1 == \"rm\" { k = $2; delete c[k] }\nEND { print c[\"keep\"] }\n",
        "keep\nkeep\nrm keep\nkeep\n", Out),
    assertion(Out == "1\n"), !.

% Delete and re-increment the same key in one guarded rule body.
test(varkey_delete_then_reinc, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkd3',
        "{ k = $1; freq[k]++ }\n$1 == \"drop\" { k = $1; delete freq[k]; freq[k]++ }\nEND { print freq[\"a\"], freq[\"drop\"] }\n",
        "a\ndrop\na\ndrop\n", Out),
    assertion(Out == "2 1\n"), !.

% Count, delete, and re-count the same key within a single unguarded rule.
test(varkey_delete_single_rule, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkd4',
        "{ k = $1; c[k]++; delete c[k]; c[k]++ } END { print c[\"a\"] }\n",
        "a\na\na\n", Out),
    assertion(Out == "1\n"), !.

% A numeric/counter scalar key for delete is a clean not-yet: no i64->string key
% path, so the program declines with a compile error rather than mis-lowering.
test(varkey_delete_numeric_rejected, [condition(clang_available)]) :-
    sdir(Dir),
    build_status(Dir, 'vkdnum',
        "{ n++; c[n]++; delete c[n] } END { print c[\"1\"] }\n", St),
    assertion(St == 3), !.

% --- read arr[k] as a value -------------------------------------------------

test(varkey_read_parses) :-
    plawk_parse_string("{ k = $1; print a[k] }\n",
        program([], [rule(always,
            [set(var(k), field(1)), print([assoc(var(a), var(k))])])], [])),
    !.

% `print c[k]` prints the running count for the current key each record.
test(varkey_read_running_count, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkr1',
        "{ k = $1; c[k]++; print c[k] } END { print c[\"a\"] }\n",
        "a\nb\na\n", Out),
    assertion(Out == "1\n1\n2\n2\n"), !.

% The key scalar stays a strnum slot: it is both printed as text (`print w`) and
% used as a key (`f[w]++`, `print f[w]`) in the same record.
test(varkey_read_key_and_count, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkr2',
        "{ w = $1; f[w]++; print w, f[w] } END { print f[\"the\"] }\n",
        "the\ncat\nthe\n", Out),
    assertion(Out == "the 1\ncat 1\nthe 2\n2\n"), !.

% The read composes with another field in a comma list.
test(varkey_read_other_field, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkr3',
        "{ k = $1; c[k]++; print $2, c[k] } END { print c[\"x\"] }\n",
        "x a\ny a\nx b\n", Out),
    assertion(Out == "a 1\na 1\nb 2\n2\n"), !.

% The read composes inside a string concatenation.
test(varkey_read_concat, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'vkr4',
        "{ k = $1; c[k]++; print \"n=\" c[k] } END { print c[\"a\"] }\n",
        "a\na\n", Out),
    assertion(Out == "n=1\nn=2\n2\n"), !.

% Reading via `printf` is a clean not-yet (declines rather than mis-lowering).
test(varkey_read_printf_rejected, [condition(clang_available)]) :-
    sdir(Dir),
    build_status(Dir, 'vkrpf',
        "{ k = $1; c[k]++; printf \"%d\\n\", c[k] } END { print c[\"a\"] }\n", St),
    assertion(St == 3), !.

% A numeric/counter scalar key read declines cleanly (no i64->string key path).
test(varkey_read_numeric_rejected, [condition(clang_available)]) :-
    sdir(Dir),
    build_status(Dir, 'vkrnum',
        "{ n++; c[n]++; print c[n] } END { print c[\"1\"] }\n", St),
    assertion(St == 3), !.

:- end_tests(plawk_assoc_varkey).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_varkey', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build only; return the compiler exit status (for clean-decline tests).
build_status(Dir, Name, Src, Status) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

build_run(Dir, Name, Src, Input, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).
