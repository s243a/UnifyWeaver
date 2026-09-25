:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% BEGIN statement separators: `;` OR newline, and a harmless trailing separator.
%
% BEGIN accepted `;` only, and committed before parsing the next statement, so the
% everyday multi-line form
%
%     BEGIN {
%         FS = ":"
%         OFS = "-"
%     }
%
% and even `BEGIN { FS = ":"; }` were parse errors (exit 2). BEGIN now uses the same
% separator (action_sep//0) and closer (action_block_close//0) as a rule body.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_begin_statements).

% All spellings of the same two-statement BEGIN parse to the same AST.
test(separators_are_equivalent) :-
    Expected = program([begin([set(var('FS'), string(":")),
                               set(var('OFS'), string("-"))])],
                       [rule(always, [print([field(1), field(2)])])], []),
    forall(member(Src,
            [ "BEGIN { FS = \":\"; OFS = \"-\" }\n{ print $1, $2 }\n",
              "BEGIN {\n  FS = \":\"\n  OFS = \"-\"\n}\n{ print $1, $2 }\n",
              "BEGIN { FS = \":\"; OFS = \"-\"; }\n{ print $1, $2 }\n",
              "BEGIN {\n  FS = \":\"   # fields\n\n  OFS = \"-\"\n}\n{ print $1, $2 }\n"
            ]),
        ( plawk_parse_string(Src, P), assertion(P == Expected) )),
    !.

test(trailing_semicolon_single_statement) :-
    plawk_parse_string("BEGIN { FS = \":\"; }\n{ print $1 }\n",
        program([begin([set(var('FS'), string(":"))])], _, [])),
    !.

test(multiline_begin_runs, [condition(clang_available)]) :-
    run("a:b:c\nx:y:z\n", "BEGIN {\n  FS = \":\"\n  OFS = \"-\"\n}\n{ print $1, $3 }\n",
        "a-c\nx-z\n"),
    !,
    run("a:b:c\nx:y:z\n", "BEGIN { FS = \":\"; }\n{ print $2 }\n", "b\ny\n"),
    !.

:- end_tests(plawk_begin_statements).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_begin_statements', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'bs_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bs', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(BS)), stderr(std), process(BPid)]),
    read_string(BS, _, _), close(BS),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).
