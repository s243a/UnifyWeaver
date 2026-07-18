:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% NR as a special in an if / while comparison guard. Previously `if (NR > 1)`
% parsed NR as a bare identifier -> a phantom scalar slot (always 0), so the
% guard was always false. Now NR in a condition is recognised as the record
% counter (special('NR')), matching the print/expr contexts. This also makes the
% record-gated idioms (`if (NR == 1) { first = ... }`, conditional accumulation)
% behave correctly -- the nested assignment itself was never broken, it was only
% ever reached on the false branch because the NR guard misread.
% (NF as a guard special is a documented follow-on: it needs the field separator
% threaded into the condition lowering and has no value in an END condition.)

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_nr_guard).

% --- parsing ---------------------------------------------------------------

test(nr_guard_parses_as_special) :-
    plawk_parse_string("{ if (NR >= 1) print 1 }\n",
        program(_, [rule(always, [if(scalar_if(Cond), _Then, _Else)])], _)),
    Cond == cmp(special('NR'), ge, int(1)),
    !.

% a variable whose name merely starts with NR is still an identifier.
test(nrx_still_identifier) :-
    plawk_parse_string("{ if (NRx > 1) print 1 }\n",
        program(_, [rule(always, [if(scalar_if(Cond), _Then, _Else)])], _)),
    Cond == cmp(var('NRx'), gt, int(1)),
    !.

% --- runtime ---------------------------------------------------------------

% NR >= 1 is true on every record (the guard reads the counter, not a 0 slot).
test(nr_ge_1_all, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nge', "{ if (NR >= 1) print \"H\"; else print \"-\" }\n",
        "x\ny\n", Out, St),
    assertion(St == 0), assertion(Out == "H\nH\n"), !.

% select a specific record by number.
test(nr_eq_selects_record, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'neq', "{ if (NR == 2) print \"second\" }\n",
        "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "second\n"), !.

% ordering comparison against the counter.
test(nr_lt, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nlt', "{ if (NR < 2) print \"early\"; else print \"late\" }\n",
        "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "early\nlate\nlate\n"), !.

% NR in an && combination (a record range).
test(nr_range_and, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nrg', "{ if (NR >= 2 && NR <= 3) print \"mid\" }\n",
        "a\nb\nc\nd\n", Out, St),
    assertion(St == 0), assertion(Out == "mid\nmid\n"), !.

% a block gated by NR, whose assignment is used AFTER the block -- the value
% propagates (nested-assign-then-use works; only the NR guard was wrong before).
test(nr_gated_block_use_after, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ngb', "{ if (NR >= 1) { c = 5 }; print c }\n",
        "x\n", Out, St),
    assertion(St == 0), assertion(Out == "5\n"), !.

% a value captured on a specific record persists to later records.
test(nr_capture_persists, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'ncp', "{ if (NR == 2) { c = 9 }; print c }\n",
        "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "0\n9\n9\n"), !.

% conditional accumulation over records (sum of positive fields).
test(nr_conditional_accumulate, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nca', "{ if ($1 > 0) { pos = pos + $1 } } END { print pos }\n",
        "3\n-1\n5\n", Out, St),
    assertion(St == 0), assertion(Out == "8\n"), !.

% NR still works in a print field (regression).
test(nr_in_print_unchanged, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nrp', "{ print NR, $0 }\n", "a\nb\n", Out, St),
    assertion(St == 0), assertion(Out == "1 a\n2 b\n"), !.

:- end_tests(plawk_nr_guard).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_nr_guard', Dir),
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
