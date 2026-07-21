:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Main-input getline shares the record driver's stream. Both target forms
% advance NR/FNR on success; only the record form replaces $0 and therefore
% changes the lazily projected fields and NF. EOF and read errors preserve the
% target and return 0/-1 respectively.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../src/unifyweaver/targets/wam_llvm_target',
              [write_wam_llvm_project/3]).

:- dynamic user:getline_main_runtime_probe/0.

user:getline_main_runtime_probe.

:- begin_tests(plawk_getline_main).

% --- parsing ---------------------------------------------------------------

test(main_record_bare_parses) :-
    plawk_parse_string("{ getline }\n",
        program([], [rule(always, [getline_main_record])], [])),
    !.

test(main_var_bare_parses) :-
    plawk_parse_string("{ getline line }\n",
        program([], [rule(always, [getline_main_var(line)])], [])),
    !.

test(main_record_capture_parses) :-
    plawk_parse_string("{ status = getline; print status }\n",
        program([], [rule(always,
            [getline_main_record_capture(status), print([var(status)])])], [])),
    !.

test(main_var_capture_parses) :-
    plawk_parse_string("{ status = getline line; print status, line }\n",
        program([], [rule(always,
            [getline_main_var_capture(status, line),
             print([var(status), var(line)])])], [])),
    !.

test(main_record_while_normalises) :-
    plawk_parse_string("{ while ((getline) > 0) print $0 }\n",
        program(_, [rule(always, Actions)], _)),
    Actions = [ getline_main_record_capture('$getline_status'),
                while_loop(cmp(var('$getline_status'), gt, int(0)),
                    [print([field(0)]),
                     getline_main_record_capture('$getline_status')]) ],
    !.

test(main_var_while_normalises) :-
    plawk_parse_string("{ while ((getline line) > 0) print line }\n",
        program(_, [rule(always, Actions)], _)),
    Actions = [ getline_main_var_capture('$getline_status', line),
                while_loop(cmp(var('$getline_status'), gt, int(0)),
                    [print([var(line)]),
                     getline_main_var_capture('$getline_status', line)]) ],
    !.

% A newline terminates the no-target form rather than being consumed as the
% whitespace before a variable target.
test(main_record_newline_boundary) :-
    plawk_parse_string("{ getline\nprint $0 }\n",
        program([], [rule(always,
            [getline_main_record, print([field(0)])])], [])),
    !.

test(main_record_capture_newline_boundary) :-
    plawk_parse_string("{ status = getline\nprint status }\n",
        program([], [rule(always,
            [getline_main_record_capture(status), print([var(status)])])], [])),
    !.

% --- runtime ---------------------------------------------------------------

% The first outer record is NR 1. Main getline consumes record 2 and replaces
% $0; the outer loop then consumes record 3, whose getline sees EOF. This also
% checks that EOF preserves the record and that END sees all three records.
test(main_record_updates_fields_counters_and_eof, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "BEGIN { FS = \":\"; OFS = \"|\" } { r = getline; print r, NR, FNR, $0, $1, $2, NF } END { print \"end\", NR, FNR }\n",
    build_run(Dir, 'record_state', Src,
        "alpha:one\nbeta:two\ngamma:three\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|2|2|beta:two|beta|two|2\n0|3|3|gamma:three|gamma|three|2\nend|3|3\n"),
    !.

% Scalar-target getline advances the counters and target, but leaves the outer
% record and its fields untouched. EOF preserves both the scalar and $0.
test(main_var_preserves_record_and_eof_target, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "BEGIN { FS = \":\"; OFS = \"|\" } { r = getline v; print r, v, NR, FNR, $0, $1, $2, NF } END { print \"end\", NR }\n",
    build_run(Dir, 'var_state', Src,
        "outer:one\ninner:two\ntail:three\n", Out, St),
    assertion(St == 0),
    assertion(Out == "1|inner:two|2|2|outer:one|outer|one|2\n0|inner:two|3|3|tail:three|tail|three|2\nend|3\n"),
    !.

test(main_record_bare_reads, [condition(clang_available)]) :-
    mdir(Dir),
    build_run(Dir, 'record_bare',
        "BEGIN { OFS = \"|\" } { getline; print NR, FNR, $0 }\n",
        "first\nsecond\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|2|second\n"),
    !.

test(main_var_bare_reads, [condition(clang_available)]) :-
    mdir(Dir),
    build_run(Dir, 'var_bare',
        "BEGIN { OFS = \"|\" } { getline v; print NR, FNR, v, $0 }\n",
        "first\nsecond\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|2|second|first\n"),
    !.

% A later guard in the same rule observes the counter and record advanced by
% getline inside an earlier branch; the hidden NR/FNR value must cross the if
% join rather than falling back to the outer-loop entry count.
test(main_record_updates_later_guard, [condition(clang_available)]) :-
    mdir(Dir),
    build_run(Dir, 'later_guard',
        "BEGIN { OFS = \"|\" } { if (NR == 1) getline; if (NR == 2) print NR, FNR, $0 }\n",
        "first\nsecond\nthird\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|2|second\n"),
    !.

test(main_record_while_drains_stream, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "BEGIN { OFS = \"|\" } { while ((getline) > 0) print NR, $0; print \"done\", NR, $0 } END { print \"end\", NR }\n",
    build_run(Dir, 'record_while', Src,
        "header\nred\ngreen\nblue\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|red\n3|green\n4|blue\ndone|4|blue\nend|4\n"),
    !.

test(main_var_while_drains_without_changing_record, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "BEGIN { OFS = \"|\" } { while ((getline v) > 0) print NR, v, $0; print \"done\", NR, v, $0 }\n",
    build_run(Dir, 'var_while', Src,
        "header\nred\ngreen\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2|red|header\n3|green|header\ndone|3|green|header\n"),
    !.

% Main getline is sourced through the same persistent reader as the outer
% driver, so it observes the active regex RS and updates RT to its own match.
test(main_record_uses_active_regex_rs, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "BEGIN { RS = \"[0-9]+\"; OFS = \"|\" } { r = getline; print r, NR, $0, RT } END { print \"end\", NR }\n",
    build_run(Dir, 'regex_rs', Src,
        "outer12inner345tail", Out, St),
    assertion(St == 0),
    assertion(Out == "1|2|inner|345\n0|3|tail|\nend|3\n"),
    !.

% A malformed handle exercises the runtime read-error path directly. Both
% helpers return -1, and the scalar helper leaves its output slot untouched.
test(main_helpers_return_minus_one_on_read_error, [condition(clang_available)]) :-
    mdir(Dir),
    directory_file_path(Dir, 'getline_main_error.ll', LLPath),
    write_wam_llvm_project(
        [user:getline_main_runtime_probe/0],
        [module_name('getline_main_error')], LLPath),
    read_file_to_string(LLPath, RuntimeIR, []),
    assertion(once(sub_string(RuntimeIR, _, _, _,
        '%gmr.line = call %Value @wam_stream_read_line_transient_value'))),
    setup_call_cleanup(
        open(LLPath, append, S, [encoding(utf8)]),
        write(S,
'\n@.getline_main_keep = private constant [5 x i8] c"keep\\00"

define i32 @main() {
entry:
  %slot = alloca i64, align 8
  store i64 777, i64* %slot
  %keep_ptr = getelementptr [5 x i8], [5 x i8]* @.getline_main_keep, i64 0, i64 0
  %seed = call i64 @wam_transient_atom_from_bytes(i8* %keep_ptr, i64 4)
  %bad0 = insertvalue %Value undef, i32 0, 0
  %bad = insertvalue %Value %bad0, i64 0, 1
  %var_status = call i64 @wam_getline_main_var(%Value %bad, i64* %slot)
  %after = load i64, i64* %slot
  %var_error = icmp eq i64 %var_status, -1
  %preserved = icmp eq i64 %after, 777
  %record_status = call i64 @wam_getline_main_record(%Value %bad)
  %record_error = icmp eq i64 %record_status, -1
  %record_ptr = call i8* @wam_atom_to_string(i64 4611686018427387904)
  %record_cmp = call i32 @strcmp(i8* %record_ptr, i8* %keep_ptr)
  %record_preserved = icmp eq i32 %record_cmp, 0
  %both0 = and i1 %var_error, %preserved
  %both1 = and i1 %both0, %record_error
  %both = and i1 %both1, %record_preserved
  %exit = select i1 %both, i32 0, i32 1
  ret i32 %exit
}
'),
        close(S)),
    directory_file_path(Dir, 'getline_main_error_bin', Bin),
    process_create(path(clang), ['-w', LLPath, '-o', Bin, '-lm'],
        [stdout(null), stderr(std), process(CPid)]),
    process_wait(CPid, exit(0)),
    process_create(Bin, [],
        [stdout(null), stderr(std), process(RPid)]),
    process_wait(RPid, exit(0)),
    !.

% --- explicit v1 rejection boundary ---------------------------------------

test(unsupported_getline_shapes_exit_3, [condition(clang_available)]) :-
    mdir(Dir),
    Cases = [ pipe-"{ cmd | getline }\n",
              dynamic_file-"{ getline < filename }\n",
              dynamic_field-"{ getline < $1 }\n",
              dynamic_scalar_target-"{ getline v < $1 }\n",
              begin_record-"BEGIN { getline } { print $0 }\n",
              begin_var-"BEGIN { getline v } { print $0 }\n",
              begin_record_capture-"BEGIN { s = getline } { print $0 }\n",
              begin_var_capture-"BEGIN { s = getline v } { print $0 }\n",
              end_record-"{ print $0 } END { getline }\n",
              end_var-"{ print $0 } END { getline v }\n",
              end_record_capture-"{ print $0 } END { s = getline }\n",
              end_var_capture-"{ print $0 } END { s = getline v }\n"
            ],
    forall(member(Name-Src, Cases),
        ( build_status(Dir, Name, Src, Status),
          assertion(Status == exit(3))
        )),
    !.

:- end_tests(plawk_getline_main).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

mdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_getline_main', Dir),
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
