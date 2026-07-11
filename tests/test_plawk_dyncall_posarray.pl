:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Positional-array target (JIT roadmap item 4's LAST target container):
% a grammar returns a FLAT list [V1, ..., Vn] and plawk lands each Vi at
% key i (1-indexed, the awk `split` convention) into one i64 array value.
%
%     arr = dyncall@fields($1) as array         % named entry
%     arr = dyncall($1) as array                % DYNLOAD default entry
%
% Same i64 table as `as assoc`, but filled from a flat list instead of
% key-value pairs (@wam_object_call_posarray walks the list into keys
% 1..n via @wam_assoc_i64_set -- REPLACE semantics, so the array reflects
% the most recent record's list). END `arr[1]`, `arr[2]`, ... and for-in
% read the positional values.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% split a number N into a 3-element positional list [N, N*10, N*100]
splitc(X, R) :-
    atom_number(X, N),
    R1 is N * 10,
    R2 is N * 100,
    R = [N, R1, R2].

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_posarray).

% `... as array` parses to a dynposarray_bind action (named + default).
test(posarray_named_parses) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         { arr = dyncall@splitc($1) as array }\n\c
         END { print arr[1], arr[2], arr[3] }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(dynposarray_bind(var(arr),
        dyncall_named(splitc, [field(1)])), Actions),
    !.

test(posarray_default_parses) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         { arr = dyncall($1) as array }\n\c
         END { print arr[1] }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(dynposarray_bind(var(arr), dyncall([field(1)])), Actions),
    !.

% `... as assoc` still parses unchanged (regression: the two targets are
% disjoint keywords after `as`).
test(assoc_still_parses) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         { arr = dyncall@splitc($1) as assoc }\n\c
         END { print arr[\"k\"] }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(dynassoc_bind(var(arr),
        dyncall_named(splitc, [field(1)])), Actions),
    !.

% NAMED entry, end to end: each record repopulates arr[1..3] positionally;
% END reads them back. Last record is 7 -> arr = [7, 70, 700].
test(posarray_named_running, [condition(clang_available)]) :-
    pa_dir(Dir),
    directory_file_path(Dir, 'splitc.wamo', Wamo),
    write_wam_object([user:splitc/2], [wamo_entries([splitc/2])], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { arr = dyncall@splitc($1) as array }\n\c
         END { print arr[1], arr[2], arr[3] }\n", [Wamo]),
    build_run(Dir, 'pan', Src, "3\n7\n", Out),
    assertion(Out == "7 70 700\n"),
    !.

% DEFAULT entry (the DYNLOAD object's wamo_entry), end to end. One record
% 5 -> arr = [5, 50, 500].
test(posarray_default_running, [condition(clang_available)]) :-
    pa_dir(Dir),
    directory_file_path(Dir, 'splitd.wamo', Wamo),
    write_wam_object([user:splitc/2], [wamo_entry(splitc/2)], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { arr = dyncall($1) as array }\n\c
         END { print arr[1], arr[2], arr[3] }\n", [Wamo]),
    build_run(Dir, 'pad', Src, "5\n", Out),
    assertion(Out == "5 50 500\n"),
    !.

% for-in over a positional array inside the rule body: iterate the slots
% the grammar just filled. One record 2 -> arr = [2, 20, 200]; the loop
% prints "pos value" per slot (slot order), END reads arr[2].
test(posarray_forin_rule_body, [condition(clang_available)]) :-
    pa_dir(Dir),
    directory_file_path(Dir, 'splitf.wamo', Wamo),
    write_wam_object([user:splitc/2], [wamo_entries([splitc/2])], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { arr = dyncall@splitc($1) as array ; for (k in arr) print k, arr[k] }\n\c
         END { print arr[2] }\n", [Wamo]),
    build_run(Dir, 'paf', Src, "2\n", Out),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, Sorted),
    assertion(Sorted == ["1 2", "2 20", "20", "3 200"]),
    !.

:- end_tests(plawk_dyncall_posarray).

% --- helpers ---------------------------------------------------------------

pa_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_posarray', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, InputText, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog0, '_in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, InputText), close(SI)),
    run_bin(Bin, [Input], Out, 0).

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
