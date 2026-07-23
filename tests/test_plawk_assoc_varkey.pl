:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Scalar-variable keys in associative arrays: `arr[x]++` where `x` is a scalar
% variable (the group-by-a-copied-field idiom, `{ k = $1; freq[k]++ }`). v1
% covers the INC operation in the mixed chain (a program whose END is a plain
% print, so scalar slots are threaded to the assoc-key site): the key scalar
% must be a string/strnum slot -- a field-assigned scalar (`k = $1`), whose slot
% value already IS the interned atom id `arr[$1]` would produce, so `arr[k]` uses
% it directly (no field-slice / re-intern). Reading the scalar as an assoc key
% is a supported strnum read, so `k` stays a strnum slot rather than deactivating
% to a numeric counter.
%
% A numeric/counter scalar key (`{ n++; arr[n]++ }`) is a clean not-yet (no
% i64->string key path), and delete/read by a scalar key are separate follow-ons.

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
