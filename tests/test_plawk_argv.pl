:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% ARGC / ARGV[N] -- the command-line arguments of the compiled program. The
% plawk driver is a native binary, so ARGV maps directly to the process argv
% read from /proc/self/cmdline: ARGV[0] is the program, ARGV[1] the first
% argument (the input path -- "-" selects stdin), ARGV[2..] any extras. ARGC is
% the count. v1 surface: rule bodies only -- ARGC as an i64 (print/guard/
% arithmetic/scalar-assign), ARGV[N] as a string scalar (print / `x = ARGV[N]`
% then string-compare). BEGIN/END prints, a direct ARGV[N] comparison operand,
% and a variable/expression index are follow-ons.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_argv).

% --- parsing ---------------------------------------------------------------

test(argv_print_parses) :-
    plawk_parse_string("{ print ARGV[1] }\n",
        program([], [rule(always, [print([argv_at(1)])])], [])),
    !.

test(argv_assign_parses) :-
    plawk_parse_string("{ x = ARGV[2]; print x }\n",
        program([], [rule(always, [set(var(x), argv_at(2)), print([var(x)])])], [])),
    !.

test(argc_print_parses) :-
    plawk_parse_string("{ print ARGC }\n",
        program([], [rule(always, [print([special('ARGC')])])], [])),
    !.

% --- runtime ---------------------------------------------------------------
% The binary is invoked with a leading "-" (stdin) so the driver reads the
% piped record; extra args land in ARGV[2..], and ARGC counts them all.

% ARGC counts the process arguments (bin, "-", foo, bar -> 4).
test(argc_count, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'ac', "{ print ARGC }\n",
        "t\n", ['-', foo, bar], Out, St),
    assertion(St == 0), assertion(Out == "4\n"), !.

% ARGV[0] is the program path (ends in the built binary name).
test(argv0_is_program, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'a0', "{ print ARGV[0] }\n",
        "t\n", ['-'], Out, St),
    assertion(St == 0),
    assertion(sub_string(Out, _, _, _, "a0_bin")), !.

% ARGV[1] is the first argument (here the input selector "-").
test(argv1_first_arg, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'a1', "{ print ARGV[1] }\n",
        "t\n", ['-', foo], Out, St),
    assertion(St == 0), assertion(Out == "-\n"), !.

% ARGV[2] is the first extra argument.
test(argv2_extra_arg, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'a2', "{ print ARGV[2] }\n",
        "t\n", ['-', hello], Out, St),
    assertion(St == 0), assertion(Out == "hello\n"), !.

% an out-of-range index is the empty string (assign-then-concat; ARGV[N]
% directly inside a juxtaposition concat parses as an assoc read -- a follow-on).
test(argv_out_of_range_empty, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'ao', "{ x = ARGV[5]; print \"[\" x \"]\" }\n",
        "t\n", ['-'], Out, St),
    assertion(St == 0), assertion(Out == "[]\n"), !.

% `x = ARGV[N]` binds a string scalar that prints and string-compares.
test(argv_assign_and_compare, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'aa',
        "{ x = ARGV[2]; if (x == \"foo\") print \"yes\"; else print \"no\" }\n",
        "t\n", ['-', foo], Out, St),
    assertion(St == 0), assertion(Out == "yes\n"), !.

% ARGC as a guard operand: `if (ARGC > 2)`.
test(argc_guard, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'ag',
        "{ if (ARGC > 2) print \"many\"; else print \"few\" }\n",
        "t\n", ['-', foo], Out, St),
    assertion(St == 0), assertion(Out == "many\n"), !.

% ARGC in arithmetic and concat.
test(argc_arith_concat, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_args(Dir, 'ar', "{ print \"n=\" (ARGC - 1) }\n",
        "t\n", ['-', foo], Out, St),
    assertion(St == 0), assertion(Out == "n=2\n"), !.

:- end_tests(plawk_argv).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_argv', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run_args(Dir, Name, Src, Input, Args, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, Args,
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
