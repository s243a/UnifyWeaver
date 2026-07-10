:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Rule-body for-in: per-record iteration over an assoc table -- e.g. one
% a grammar just populated via `arr = dyncall@t($1) as assoc` -- printing
% one line per key, inside the rule's action chain rather than only in
% END. The loop sees the table AS ACCUMULATED SO FAR (a running
% snapshot per record).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% atom-keyed tally: every record bumps seen, big/small bucket by size
tallyc(X, R) :-
    atom_number(X, N),
    ( N > 100 -> R = [big-2, seen-1] ; R = [small-1, seen-1] ).

% str-valued labeler (the assoc(str) table kind)
labeler(X, R) :-
    atom_number(X, N),
    ( N > 100 -> R = [size-big, last-X] ; R = [size-small, last-X] ).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_forin_rule_body).

% for (k in arr) parses as a rule-body ACTION (same node as the END
% form, inside the rule's action list).
test(rule_body_forin_parses) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         { arr = dyncall@tallyc($1) as assoc ; for (k in arr) print k, arr[k] }\n\c
         END { print arr[\"seen\"] }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(for_in(var(k), var(arr),
        [print([var(k), assoc(var(arr), var(k))])]), Actions),
    !.

% Per-record snapshot: each record populates then iterates the RUNNING
% table. 50 -> {small:1, seen:1} (2 lines); 200 -> {small:1, seen:2,
% big:2} (3 lines); END reads seen=2. Line order is slot order, so
% compare sorted.
test(rule_body_forin_running_snapshot, [condition(clang_available)]) :-
    fr_dir(Dir),
    directory_file_path(Dir, 'tallyc.wamo', Wamo),
    write_wam_object([user:tallyc/2], [wamo_entries([tallyc/2])], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { arr = dyncall@tallyc($1) as assoc ; for (k in arr) print k, arr[k] }\n\c
         END { print arr[\"seen\"] }\n", [Wamo]),
    build_run(Dir, 'frb', Src, "50\n200\n", Out),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, Sorted),
    assertion(Sorted ==
        ["2", "big 2", "seen 1", "seen 2", "small 1", "small 1"]),
    !.

% str-valued tables print their values as TEXT inside the rule loop too
% (the planned field carries the table's value kind). One record, two
% keys; last record 7 -> size small, last 7.
test(rule_body_forin_str_values, [condition(clang_available)]) :-
    fr_dir(Dir),
    directory_file_path(Dir, 'labeler.wamo', Wamo),
    write_wam_object([user:labeler/2], [wamo_entries([labeler/2])], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         { arr = dyncall@labeler($1) as assoc(str) ; for (k in arr) print k, arr[k], \"row\" }\n\c
         END { print arr[\"size\"] }\n", [Wamo]),
    build_run(Dir, 'frs', Src, "7\n", Out),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, Sorted),
    assertion(Sorted == ["last 7 row", "size small row", "small"]),
    !.

:- end_tests(plawk_forin_rule_body).

% --- helpers ---------------------------------------------------------------

fr_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_forin_rule', Dir),
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
