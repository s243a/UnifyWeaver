:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_cli, [condition(clang_available)]).

test(build_text_program_reads_file_arg_and_stdin) :-
    cli_dir(Dir),
    write_text_file(Dir, 'errors.plawk',
"# count ERROR lines\n$1 == \"ERROR\" { errors++ }\n{ total++ }\nEND { print errors, total }\n"),
    write_text_file(Dir, 'input.txt',
"INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n"),
    directory_file_path(Dir, 'errors.plawk', Prog),
    directory_file_path(Dir, 'errors_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'input.txt', Input),
    run_capture(Bin, [Input], FileOut, 0),
    assertion(FileOut == "2 4\n"),
    run_capture_stdin(Bin, Input, StdinOut, 0),
    assertion(StdinOut == "2 4\n"),
    !.

test(build_with_prolog_block_and_function) :-
    cli_dir(Dir),
    write_text_file(Dir, 'weights.plawk',
"@prolog\nplawk_clit_hot(X) :- X > 100.\n@end\n\nBEGIN { BINFMT = \"i64 f64\" }\nfunction plawk_clit_scale(a) { return a * 2 }\nplawk_clit_hot($1) { wsum += float($2) ; s += plawk_clit_scale($1) }\nEND { print s, wsum }\n"),
    directory_file_path(Dir, 'weights.plawk', Prog),
    directory_file_path(Dir, 'weights_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'weights.bin', Input),
    write_weights_records(Input, [rec(200, 1.5), rec(5, 2.5), rec(300, 0.25)]),
    run_capture(Bin, [Input], Out, 0),
    assertion(Out == "1000 1.75\n"),
    !.

test(run_mode_builds_and_executes_with_clean_stdout) :-
    cli_dir(Dir),
    directory_file_path(Dir, 'errors.plawk', Prog),
    directory_file_path(Dir, 'input.txt', Input),
    cli([run, Prog, Input], Out, 0),
    assertion(Out == "2 4\n"),
    !.

test(parse_error_exits_2) :-
    cli_dir(Dir),
    write_text_file(Dir, 'bad.plawk', "{ n++ n++ }\nEND { print n }\n"),
    directory_file_path(Dir, 'bad.plawk', Prog),
    cli([build, Prog, '-o', '/dev/null'], _, 2),
    !.

test(missing_bridged_predicate_exits_3) :-
    cli_dir(Dir),
    write_text_file(Dir, 'missing.plawk',
"{ total += nosuch($1) }\nEND { print total }\n"),
    directory_file_path(Dir, 'missing.plawk', Prog),
    cli([build, Prog, '-o', '/dev/null'], _, 3),
    !.

:- end_tests(plawk_cli).

% --- helpers ---------------------------------------------------------------

cli_dir(Dir) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_cli', Dir),
    (   exists_directory(Dir)
    ->  true
    ;   make_directory_path(Dir)
    ).

write_text_file(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(
        open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text),
        close(Out)).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

double_bits(1.5,  0x3FF8000000000000).
double_bits(2.5,  0x4004000000000000).
double_bits(0.25, 0x3FD0000000000000).

write_weights_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(I, F), Recs),
            ( write_i64_le(Out, I),
              double_bits(F, Bits),
              write_i64_le(Out, Bits) )),
        close(Out)).

% invoke the CLI script through swipl; capture stdout + exit status
cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(null), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_capture(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_capture_stdin(Bin, InputPath, Out, ExpectedStatus) :-
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(S)), stderr(std), process(Pid)]),
    setup_call_cleanup(
        open(InputPath, read, F, [type(binary)]),
        copy_stream_data(F, In),
        close(F)),
    close(In),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
