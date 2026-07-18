:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Per-record printing from a rule body that also mutates an assoc/positional
% table, with NO END block: `{ c[$1]++; print $1, c[$1] }` (a running count) and
% `{ split($0,a,","); print a[1] }` (print an element). Previously an assoc
% program had to be END-tethered -- a rule that mutated a table AND printed
% per-record was rejected ("outside the compilable surface") unless the program
% also had an `END { print arr[...] }`. The per-record print machinery (the
% assoc rule chain's assoc_print_action, with %line live) already existed; this
% exercises the no-END driver path that reaches it, plus integer-key element
% reads (`arr[N]`).
%
% Supported body-print fields: `$N` (N>0), `arr[N]` (integer element), `arr[$k]`
% (field-keyed lookup), and scalar reads. String literals and `$0` in a body
% assoc print are follow-ons.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_assoc_body_print).

% --- parsing ---------------------------------------------------------------

test(counter_body_print_parses) :-
    plawk_parse_string("{ c[$1]++; print $1, c[$1] }\n",
        program([], [rule(always,
            [inc_assoc(var(c), field(1)), print([field(1), assoc(var(c), field(1))])])], [])),
    !.

test(split_body_print_parses) :-
    plawk_parse_string("{ split($0, a, \",\"); print a[1] }\n",
        program([], [rule(always,
            [split_into(field(0), var(a), string(",")), print([assoc(var(a), int(1))])])], [])),
    !.

% --- runtime ---------------------------------------------------------------

% A running count per record: c[$1]++ then print the key and its count so far.
test(counter_running_count, [condition(clang_available)]) :-
    adir(Dir),
    build_run(Dir, 'cc', "{ c[$1]++; print $1, c[$1] }\n", "a\nb\na\n", Out),
    assertion(Out == "a 1\nb 1\na 2\n"), !.

% split the record and print the first element per record (an awk `-F, '{print $1}'`).
test(split_print_first, [condition(clang_available)]) :-
    adir(Dir),
    build_run(Dir, 'sf', "{ split($0, a, \",\"); print a[1] }\n", "x,y,z\np,q,r\n", Out),
    assertion(Out == "x\np\n"), !.

% print two elements, and in a reordered order.
test(split_print_reorder, [condition(clang_available)]) :-
    adir(Dir),
    build_run(Dir, 'sr', "{ split($0, a, \",\"); print a[3], a[1] }\n", "x,y,z\n", Out),
    assertion(Out == "z x\n"), !.

% a field-keyed lookup in the body print (arr[$k]).
test(counter_lookup_field_key, [condition(clang_available)]) :-
    adir(Dir),
    build_run(Dir, 'lf', "{ c[$1]++; print c[$1] }\n", "a\na\nb\n", Out),
    assertion(Out == "1\n2\n1\n"), !.

% a pattern guard gates the per-record print + mutation.
test(guarded_body_print, [condition(clang_available)]) :-
    adir(Dir),
    build_run(Dir, 'gp', "/a/ { c[$1]++; print $1, c[$1] }\n", "a\nb\na\n", Out),
    assertion(Out == "a 1\na 2\n"), !.

% BEGIN runs before the per-record loop.
test(begin_then_body_print, [condition(clang_available)]) :-
    adir(Dir),
    build_run(Dir, 'bg', "BEGIN { print \"start\" }\n{ c[$1]++; print c[$1] }\n",
        "a\na\n", Out),
    assertion(Out == "start\n1\n2\n"), !.

:- end_tests(plawk_assoc_body_print).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

adir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_body_print', Dir),
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
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).
