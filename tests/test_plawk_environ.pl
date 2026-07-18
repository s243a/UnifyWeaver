:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% ENVIRON["NAME"] -- read-only lookup of an environment variable (getenv), the
% value as a string. Unset -> "" (awk semantics). v1: a string-literal key, in a
% scalar assignment (`x = ENVIRON["NAME"]`, a string scalar -- so it also prints
% and string-compares) and as a direct print field (`print ENVIRON["NAME"]`).
% Assigning ENVIRON to an array or a non-literal key is a follow-on.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_environ).

% --- parsing ---------------------------------------------------------------

test(environ_assign_parses) :-
    plawk_parse_string("{ x = ENVIRON[\"HOME\"]; print x }\n",
        program([], [rule(always, [set(var(x), environ("HOME")), print([var(x)])])], [])),
    !.

test(environ_print_parses) :-
    plawk_parse_string("{ print ENVIRON[\"HOME\"] }\n",
        program([], [rule(always, [print([environ("HOME")])])], [])),
    !.

% --- runtime ---------------------------------------------------------------

% x = ENVIRON["NAME"] reads the env var value.
test(environ_assign_value, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_env(Dir, 'ea', "{ x = ENVIRON[\"PLAWK_ENV_A\"]; print x }\n",
        "t\n", ['PLAWK_ENV_A'='hello_env'], Out, St),
    assertion(St == 0), assertion(Out == "hello_env\n"), !.

% direct print of an env var.
test(environ_direct_print, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_env(Dir, 'ep', "{ print ENVIRON[\"PLAWK_ENV_A\"] }\n",
        "t\n", ['PLAWK_ENV_A'='direct_val'], Out, St),
    assertion(St == 0), assertion(Out == "direct_val\n"), !.

% an unset variable is the empty string.
test(environ_unset_empty, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_env(Dir, 'eu', "{ x = ENVIRON[\"PLAWK_ENV_UNSET\"]; print \"[\" x \"]\" }\n",
        "t\n", [], Out, St),
    assertion(St == 0), assertion(Out == "[]\n"), !.

% the env value string-compares (via the string-scalar guard).
test(environ_string_compare, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_env(Dir, 'ec',
        "{ d = ENVIRON[\"PLAWK_ENV_DEBUG\"]; if (d == \"1\") print \"on\"; else print \"off\" }\n",
        "t\n", ['PLAWK_ENV_DEBUG'='1'], Out, St),
    assertion(St == 0), assertion(Out == "on\n"), !.

% a value as one field of a comma-separated print list (juxtaposition-concat
% with ENVIRON is a follow-on; the comma list works).
test(environ_in_print_list, [condition(clang_available)]) :-
    ldir(Dir),
    build_run_env(Dir, 'ecc', "{ print \"v\", ENVIRON[\"PLAWK_ENV_A\"] }\n",
        "t\n", ['PLAWK_ENV_A'='xyz'], Out, St),
    assertion(St == 0), assertion(Out == "v xyz\n"), !.

:- end_tests(plawk_environ).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_environ', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run_env(Dir, Name, Src, Input, EnvPairs, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid),
         environment(EnvPairs)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
