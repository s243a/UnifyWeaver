:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk `split($N, arr, "sep")` -- split a field on a single-char separator into
% a positional array `arr` (keys 1..n, string values), read back via `arr[i]`
% and for-in. Backed by the @wam_str_split_into runtime primitive (clears the
% table and repopulates -- split replaces the array each call). $0 is the whole
% record; a positive field uses its slice. Empty pieces are kept (explicit
% single-char separator, as awk). v1: field source, string-literal single-char
% separator; the return count and a default (FS) separator are follow-ons.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_split).

% --- parsing ----------------------------------------------------------------

test(split_parses) :-
    plawk_parse_string("{ split($0, a, \",\") }\n",
        program([], [rule(always, [split_into(field(0), var(a), string(","))])], [])),
    !.

test(split_field_parses) :-
    plawk_parse_string("{ split($2, parts, \":\") }\n",
        program([], [rule(always, [split_into(field(2), var(parts), string(":"))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% split $0 into positions 1..n; for-in yields integer keys + string values.
test(split_record_key_value, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'kv',
        "{ split($0, a, \",\") }\nEND { for (k in a) print k, a[k] }\n",
        "apple,banana,cherry\n", Out, St),
    assertion(St == 0), assertion(Out == "1 apple\n2 banana\n3 cherry\n"), !.

% value-only read.
test(split_values, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'v', "{ split($0, a, \",\") }\nEND { for (k in a) print a[k] }\n",
        "x,y,z\n", Out, St),
    assertion(St == 0), assertion(Out == "x\ny\nz\n"), !.

% split a positive field.
test(split_field_source, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'fl', "{ split($2, a, \":\") }\nEND { for (k in a) print k, a[k] }\n",
        "ignore x:y:z\n", Out, St),
    assertion(St == 0), assertion(Out == "1 x\n2 y\n3 z\n"), !.

% empty middle piece is kept.
test(split_keeps_empty, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'em', "{ split($0, a, \",\") }\nEND { for (k in a) print k, a[k] }\n",
        "a,,b\n", Out, St),
    assertion(St == 0), assertion(Out == "1 a\n2 \n3 b\n"), !.

% a source with no separator is one piece.
test(split_single_piece, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'sg', "{ split($0, a, \",\") }\nEND { for (k in a) print k, a[k] }\n",
        "solo\n", Out, St),
    assertion(St == 0), assertion(Out == "1 solo\n"), !.

% each record's split replaces the array (the table is cleared first).
test(split_replaces_each_record, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'rp', "{ split($0, a, \",\") }\nEND { for (k in a) print a[k] }\n",
        "p,q\nx,y,z\n", Out, St),
    assertion(St == 0), assertion(Out == "x\ny\nz\n"), !.

% a leading literal label prints alongside the for-in pair.
test(split_labelled_forin, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'lb',
        "{ split($0, a, \",\") }\nEND { for (k in a) print \"col\", k, a[k] }\n",
        "x,y\n", Out, St),
    assertion(St == 0), assertion(Out == "col 1 x\ncol 2 y\n"), !.

:- end_tests(plawk_split).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_split', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
