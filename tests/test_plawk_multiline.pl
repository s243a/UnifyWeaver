:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

:- dynamic user:plawk_ml_marker/0.

user:plawk_ml_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_multiline, [condition(clang_available)]).

test(newlines_separate_statements) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" }\n{ a++\n  b++; c++\n}\nEND { print a, b, c }\n", Program),
    Program = program(_, [rule(always, Actions)], _),
    assertion(Actions == [inc(var(a)), inc(var(b)), inc(var(c))]).

test(comments_run_to_end_of_line) :-
    plawk_parse_string("# leading comment\nBEGIN { BINFMT = \"i64 f64\" }   # layout\n{ a++   # bump\n  b++ }\nEND { print a, b }   # report\n", Program),
    Program = program(_, [rule(always, Actions)], _),
    assertion(Actions == [inc(var(a)), inc(var(b))]),
    % '#' inside a string literal is not a comment
    plawk_parse_string("$1 == \"#x\" { n++ } END { print n }\n", P2),
    P2 = program(_, [rule(field_eq(1, "#x"), _)], _).

test(no_separator_needed_after_a_block) :-
    % awk/C semantics: a compound statement's closing brace ends the
    % statement, on the same line or not.
    plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" } { if ($1 > 10) { h++ } w++ } END { print h, w }\n", P1),
    P1 = program(_, [rule(always, [if(_, _, _), inc(var(w))])], _),
    plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" } { if ($1 > 10) { h++ }\n  w++ } END { print h, w }\n", P2),
    P2 = program(_, [rule(always, [if(_, _, _), inc(var(w))])], _).

test(trailing_separators_are_harmless) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" } { a++;\n } END { print a }\n", Program),
    Program = program(_, [rule(always, [inc(var(a))])], _).

test(adjacent_statements_still_need_a_separator) :-
    assertion(\+ plawk_parse_string("BEGIN { BINFMT = \"i64 f64\" } { a++ b++ } END { print a }\n", _)).

test(surface_multiline_program_end_to_end) :-
    % A real multi-line program: comments, newline separators, a
    % same-line statement after an if block, multi-line BEGIN.
    Src = "# count big and small records\nBEGIN {\n  BINFMT = \"i64 f64\"   # binary layout\n}\n$1 > 100 {\n  big++\n  if ($1 > 1000) { huge++ }\n  wsum += float($2)   # weight\n}\n{ total++ }\nEND { print big, huge, total, wsum }\n",
    run_ml_smoke(Src,
        [rec(200, 1.5), rec(5, 2.5), rec(2000, 0.25)],
        "2 1 3 1.75\n").

:- end_tests(plawk_multiline).

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% dyadic doubles used by these tests
double_bits(1.5,  0x3FF8000000000000).
double_bits(2.5,  0x4004000000000000).
double_bits(0.25, 0x3FD0000000000000).

write_ml_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(I, F), Recs),
            ( write_i64_le(Out, I),
              double_bits(F, Bits),
              write_i64_le(Out, Bits) )),
        close(Out)).

run_ml_smoke(Source, Recs, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_multiline', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_ml_marker/0 ],
        [module_name('plawk_multiline')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(BuildS)), stderr(std), process(BuildPid)]),
    read_string(BuildS, _, BuildOut),
    close(BuildS),
    process_wait(BuildPid, BuildStatus),
    ( BuildStatus == exit(0)
    -> true
    ;  format(user_error, "~n[plawk multiline build output]~n~w~n", [BuildOut]),
       throw(plawk_multiline_build_failed(BuildStatus))
    ),
    write_ml_records(InputPath, Recs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == ExpectedOutput),
    !.
