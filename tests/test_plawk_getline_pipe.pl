:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Command-pipe getline runs a literal shell command once and advances through
% its stdout through a command-keyed registry. Both target forms advance the
% logical NR counter on success (and therefore FNR, which is an NR alias in
% plawk); only the record form replaces $0 and its lazily projected fields.
% Physical newlines delimit records in v1. EOF/error preserve the destination.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

:- begin_tests(plawk_getline_pipe).

% --- parsing ---------------------------------------------------------------

test(pipe_record_bare_parses) :-
    plawk_parse_string("{ \"echo a:b\" | getline }\n",
        program([], [rule(always,
            [getline_pipe_record("echo a:b")])], [])),
    !.

test(pipe_var_bare_parses) :-
    plawk_parse_string("{ \"echo value\" | getline line }\n",
        program([], [rule(always,
            [getline_pipe_read(line, "echo value")])], [])),
    !.

test(pipe_record_capture_parses) :-
    plawk_parse_string("{ status = \"echo value\" | getline; print status }\n",
        program([], [rule(always,
            [getline_pipe_record_capture(status, "echo value"),
             print([var(status)])])], [])),
    !.

test(pipe_var_capture_parses) :-
    plawk_parse_string(
        "{ status = \"echo value\" | getline line; print status, line }\n",
        program([], [rule(always,
            [getline_pipe_capture(status, line, "echo value"),
             print([var(status), var(line)])])], [])),
    !.

test(pipe_record_while_normalises) :-
    plawk_parse_string(
        "{ while ((\"seq 3\" | getline) > 0) print $0 }\n",
        program(_, [rule(always, Actions)], _)),
    Actions = [ getline_pipe_record_capture('$getline_status', "seq 3"),
                while_loop(cmp(var('$getline_status'), gt, int(0)),
                    [print([field(0)]),
                     getline_pipe_record_capture('$getline_status',
                         "seq 3")]) ],
    !.

test(pipe_var_while_normalises) :-
    plawk_parse_string(
        "{ while ((\"seq 3\" | getline line) > 0) print line }\n",
        program(_, [rule(always, Actions)], _)),
    Actions = [ getline_pipe_capture('$getline_status', line, "seq 3"),
                while_loop(cmp(var('$getline_status'), gt, int(0)),
                    [print([var(line)]),
                     getline_pipe_capture('$getline_status', line,
                         "seq 3")]) ],
    !.

% A scalar pipe read appears inside the normalized loop body. Its scratch output
% must have stable storage rather than an alloca executed once per record; the
% latter exhausts the stack on a long drain.
test(pipe_var_while_uses_constant_stack_ir) :-
    Src = "{ while ((\"seq 3\" | getline line) > 0) print line }\n",
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '_lineid_slot = private global i64 0'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '_lineid = alloca i64')),
    !.

% --- runtime ---------------------------------------------------------------

% The record form re-splits with the active regex FS and advances the shared
% NR/FNR value. This is a bare statement, so its status is intentionally unused.
test(pipe_record_bare_resplits_and_advances_nr,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { FS = \"[,:]+\"; OFS = \"|\" } { \"echo left::right,tail\" | getline; print NR, FNR, $0, $1, $2, $3, NF }\n",
    build_run(Dir, 'record_bare', Src, "trigger\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|2|left::right,tail|left|right|tail|3\n"),
    !.

% Scalar-target pipe getline advances NR/FNR but preserves the current record,
% its fields, and NF.
test(pipe_var_bare_preserves_record, [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { FS = \":\"; OFS = \"|\" } { \"echo inner:two\" | getline v; print NR, FNR, v, $0, $1, $2, NF }\n",
    build_run(Dir, 'var_bare', Src, "outer:one\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|2|inner:two|outer:one|outer|one|2\n"),
    !.

% The command is opened once. The second read observes clean EOF, returns 0,
% does not increment NR, and leaves the successful pipe record in $0.
test(pipe_record_status_eof_preserves_record,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { FS = \":\"; OFS = \"|\" } { r = \"echo one:two\" | getline; print r, NR, $0, $1, $2, NF; r = \"echo one:two\" | getline; print r, NR, $0, $1, $2, NF }\n",
    build_run(Dir, 'record_eof', Src, "outer:main\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|2|one:two|one|two|2\n0|2|one:two|one|two|2\n"),
    !.

test(pipe_var_status_eof_preserves_target,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { OFS = \"|\" } { v = \"keep\"; r = \"echo fresh\" | getline v; print r, v, NR, $0; r = \"echo fresh\" | getline v; print r, v, NR, $0 }\n",
    build_run(Dir, 'var_eof', Src, "outer\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|fresh|2|outer\n0|fresh|2|outer\n"),
    !.

test(pipe_record_while_drains, [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { OFS = \"|\" } { while ((\"seq 3\" | getline) > 0) print NR, FNR, $0, NF; print \"done\", NR, $0 } END { print \"end\", NR, FNR }\n",
    build_run(Dir, 'record_while', Src, "header\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|2|1|1\n3|3|2|1\n4|4|3|1\ndone|4|3\nend|4|4\n"),
    !.

test(pipe_var_while_drains_without_changing_record,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { FS = \":\"; OFS = \"|\" } { while ((\"seq 3\" | getline v) > 0) print NR, FNR, v, $0, $1, $2, NF; print \"done\", NR, v, $0 }\n",
    build_run(Dir, 'var_while', Src, "outer:row\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|2|1|outer:row|outer|row|2\n3|3|2|outer:row|outer|row|2\n4|4|3|outer:row|outer|row|2\ndone|4|3|outer:row\n"),
    !.

% Record and scalar forms share one registry entry for identical command text.
% The assoc update/END lookup also forces the mixed driver, whose hidden NR
% slot must carry both successful reads through to the print and END report.
test(pipe_forms_share_command_registry, [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { OFS = \"|\" } { seen[$1]++; a = \"seq 2\" | getline v; b = \"seq 2\" | getline; print a, v, b, $0, NR, FNR } END { print seen[\"outer\"], NR }\n",
    build_run(Dir, 'shared_registry', Src, "outer\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|1|1|2|3|3\n1|3\n"),
    !.

% The pipe reader grows its line buffer and the record target refreshes direct
% $0 consumers after the transient buffer moves.
test(pipe_record_long_line, [condition(clang_available)]) :-
    pdir(Dir),
    Src = "{ \"printf %05000d 0; echo\" | getline; print $0 }\n",
    build_run(Dir, 'long_record', Src, "trigger\n", Out, St),
    assertion(St == 0),
    length(ZeroCodes, 5000),
    maplist(=(0'0), ZeroCodes),
    string_codes(Zeros, ZeroCodes),
    string_concat(Zeros, "\n", Expected),
    assertion(Out == Expected),
    !.

% Pipe getline is physical-newline-only in v1. It does not consume the active
% regex RS or replace the RT captured by the main-input reader.
test(pipe_record_ignores_rs_and_preserves_rt,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { RS = \"[0-9]+\"; OFS = \"|\" } { r = \"echo left45right\" | getline; print r, NR, $0, RT }\n",
    build_run(Dir, 'rs_rt', Src, "trigger123", Out, St),
    assertion(St == 0),
    assertion(Out == "1|2|left45right|123\n"),
    !.

% Once popen succeeds, a child that exits nonzero without output is still a
% clean pipe EOF, as in gawk. The child status is not a getline read error.
test(pipe_record_nonzero_command_is_eof,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { FS = \":\"; OFS = \"|\" } { r = \"false\" | getline; print r, NR, FNR, $0, $1, $2, NF }\n",
    build_run(Dir, 'record_error', Src, "outer:main\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0|1|1|outer:main|outer|main|2\n"),
    !.

test(pipe_var_nonzero_command_is_eof,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { OFS = \"|\" } { v = \"keep\"; r = \"false\" | getline v; print r, v, NR, FNR, $0 }\n",
    build_run(Dir, 'var_error', Src, "outer\n", Out, St),
    assertion(St == 0),
    assertion(Out == "0|keep|1|1|outer\n"),
    !.

% Hold fds 3..8 open, then limit the executed binary to fds 0..9. The dynamic
% loader can reuse fd 9, but popen cannot create its two-fd pipe. This
% deterministically exercises a real spawn/open failure: both forms return -1
% and preserve their destinations and NR/FNR.
test(pipe_spawn_error_returns_minus_one_and_preserves_targets,
        [condition(clang_available)]) :-
    pdir(Dir),
    Src = "BEGIN { FS = \":\"; OFS = \"|\" } { v = \"keep\"; a = \"echo record\" | getline; b = \"echo scalar\" | getline v; print a, b, v, NR, FNR, $0, $1, $2, NF }\n",
    build_run_low_nofile(Dir, 'spawn_error', Src, "outer:main\n", Out, St),
    assertion(St == 0),
    assertion(Out == "-1|-1|keep|1|1|outer:main|outer|main|2\n"),
    !.

% --- explicit v1 rejection boundary ---------------------------------------

test(unsupported_pipe_getline_shapes_exit_3,
        [condition(clang_available)]) :-
    pdir(Dir),
    Cases = [ dynamic_record-"{ cmd = \"echo x\"; cmd | getline }\n",
              dynamic_var-"{ cmd = \"echo x\"; cmd | getline v }\n",
              dynamic_field-"{ $1 | getline }\n",
              dynamic_paren-"{ (cmd) | getline }\n",
              capture_dynamic-"{ s = cmd | getline }\n",
              while_dynamic-"{ while ((cmd | getline) > 0) print $0 }\n",
              begin_record-"BEGIN { \"echo x\" | getline } { print $0 }\n",
              begin_var-"BEGIN { \"echo x\" | getline v } { print $0 }\n",
              begin_capture-"BEGIN { s = \"echo x\" | getline } { print $0 }\n",
              end_record-"{ print $0 } END { \"echo x\" | getline }\n",
              end_var-"{ print $0 } END { \"echo x\" | getline v }\n",
              end_capture-"{ print $0 } END { s = \"echo x\" | getline }\n"
            ],
    forall(member(Name-Src, Cases),
        ( build_status(Dir, Name, Src, Status),
          assertion(Status == exit(3))
        )),
    !.

:- end_tests(plawk_getline_pipe).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

pdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_getline_pipe', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin-Prog) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        format(S, "~s", [Src]), close(S)),
    atom_concat(Prog0, '_bin', Bin).

build_status(Dir, Name, Src, Status) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, Status).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~s", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).

build_run_low_nofile(Dir, Name, Src, Input, Out, RunStatus) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    RunCommand =
        'exec 3</dev/null 4</dev/null 5</dev/null 6</dev/null 7</dev/null 8</dev/null; ulimit -n 10; exec "$1"',
    process_create(path(sh),
        ['-c', RunCommand, sh, Bin],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~s", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
