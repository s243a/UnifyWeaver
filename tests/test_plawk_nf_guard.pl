:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% NF as a comparison special in an `if`/`while` guard: `if (NF > 3)`. NF is the
% field count of the current record, computed from %line via the field-count
% runtime with the field separator threaded into the condition lowering (the
% reason it was a follow-on after NR). It is only valid where a record is in
% scope -- an END condition (no current record) is a clean not-yet.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_nf_guard).

% --- parsing ---------------------------------------------------------------

% `if (NF > 3)` parses to a scalar_if over a special('NF') comparison, not a
% phantom scalar slot named NF.
test(nf_guard_parses) :-
    plawk_parse_string("{ if (NF > 3) print $0 }\n",
        program([], [rule(always,
            [if(scalar_if(cmp(special('NF'), gt, int(3))), [print([field(0)])], [])])], [])),
    !.

% --- runtime ---------------------------------------------------------------

% `if (NF > 2)` keeps records with more than two (whitespace) fields.
test(nf_gt, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'gt', "{ if (NF > 2) print $0 }\n", "a b c\nx y\np q r s\n", Out),
    assertion(Out == "a b c\np q r s\n"), !.

% `if (NF == 2)` -- exact field count.
test(nf_eq, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'eq', "{ if (NF == 2) print \"two:\", $0 }\n", "a b\nc\nd e\n", Out),
    assertion(Out == "two: a b\ntwo: d e\n"), !.

% NF is computed against the active FS (here a comma), not whitespace.
test(nf_with_fs, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'fs', "BEGIN { FS=\",\" }\n{ if (NF >= 3) print $1 }\n",
        "a,b,c\nx,y\np,q,r,s\n", Out),
    assertion(Out == "a\np\n"), !.

% NF combined in a boolean guard with NR.
test(nf_and_nr, [condition(clang_available)]) :-
    ndir(Dir),
    build_run(Dir, 'an', "{ if (NF >= 2 && NR == 1) print $0 }\n",
        "a b\nc d\n", Out),
    assertion(Out == "a b\n"), !.

:- end_tests(plawk_nf_guard).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ndir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_nf_guard', Dir),
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
